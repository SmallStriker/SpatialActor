import math
from functools import wraps
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn, einsum

try:
    import xformers.ops as xops
except ImportError as e:
    xops = None

LRELU_SLOPE = 0.02

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):  # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, use_fast=False, use_separate_kv=False):

        super().__init__()
        self.use_fast = use_fast
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.use_separate_kv = use_separate_kv

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        if self.use_separate_kv:
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        else:
            self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout_p = dropout
        # dropout left in use_fast for backward compatibility
        self.dropout = nn.Dropout(self.dropout_p)

        self.avail_xf = False
        if self.use_fast:
            if not xops is None:
                self.avail_xf = True
            else:
                self.use_fast = False

    def forward(self, x, context=None, context_v=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        # Compute k, v
        if self.use_separate_kv:
            assert context is not None
            assert context_v is not None
            k = self.to_k(context)
            v = self.to_v(context_v)
        else:
            context = default(context, x)
            k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        if self.use_fast:
            # using py2 if available
            dropout_p = self.dropout_p if self.training else 0.0
            # using xf if available
            if self.avail_xf:
                out = xops.memory_efficient_attention(
                    query=q, key=k, value=v, p=dropout_p
                )
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)
            # attention
            attn = sim.softmax(dim=-1)
            # dropout
            attn = self.dropout(attn)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        return out


def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    elif norm == "group":
        return nn.GroupNorm(4, num_channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super().__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        else:
            self.norm = None
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_sizes=3,
        norm=None,
        activation=None,
        out_size=None,
    ):
        super().__init__()
        layer = [
            Conv2DBlock(in_channels, out_channels, kernel_sizes, 1, norm, activation)
        ]
        if strides > 1:
            if out_size is None:
                layer.append(
                    nn.Upsample(scale_factor=strides, mode="bilinear", align_corners=False)
                )
            else:
                layer.append(
                    nn.Upsample(size=out_size, mode="bilinear", align_corners=False)
                )

        if out_size is not None:
            if kernel_sizes % 2 == 0:
                kernel_sizes += 1

        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation
        )
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


###### scene_level 稀疏注意力
class PairwiseAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, attn_mask=None):
        context = default(context, x)
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(attn_mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            attn_mask = repeat(attn_mask, "b i j -> (b h) i j", h=h)
            sim.masked_fill_(~attn_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SparseSceneReasoning(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.coarse_attn = PreNorm(
            dim,
            Attention(
                dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        self.coarse_ffn = PreNorm(dim, FeedForward(dim))
        self.fine_cross_attn = PreNorm(
            dim,
            PairwiseAttention(
                dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        self.fine_context_attn = PreNorm(
            dim,
            Attention(
                dim,
                context_dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        self.fine_ffn = PreNorm(dim, FeedForward(dim))

    @staticmethod
    def _gather_tokens(x, idx):
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, 1, gather_idx)

    @staticmethod
    def _scatter_tokens(base, idx, update):
        scatter_idx = idx.unsqueeze(-1).expand(-1, -1, base.shape[-1])
        return base.scatter(1, scatter_idx, update)

    @staticmethod
    def _coarse_stride(h, w, keep_ratio):
        if min(h, w) >= 16 and keep_ratio <= 0.25:
            return 4
        if min(h, w) >= 8 and keep_ratio <= 0.5:
            return 2
        return 1

    @staticmethod
    def _build_pairwise_mask(fine_xyz, fine_views, corr_radius, corr_topk):
        bsz, k, _ = fine_xyz.shape
        dist = torch.cdist(fine_xyz.float(), fine_xyz.float())
        cross_view = fine_views.unsqueeze(-1) != fine_views.unsqueeze(1)
        radius_mask = dist <= corr_radius

        topk = min(corr_topk, max(k - 1, 1))
        masked_dist = dist.masked_fill(~cross_view, float("inf"))
        nn_idx = masked_dist.topk(topk, dim=-1, largest=False).indices
        topk_mask = torch.zeros_like(cross_view)
        topk_mask.scatter_(2, nn_idx, True)

        pairwise_mask = cross_view & (radius_mask | topk_mask)
        eye = torch.eye(k, device=fine_xyz.device, dtype=torch.bool).unsqueeze(0)
        pairwise_mask = pairwise_mask | eye
        no_neighbor = ~pairwise_mask.any(dim=-1, keepdim=True)
        pairwise_mask = pairwise_mask | (eye & no_neighbor)
        return pairwise_mask

    def forward(
        self,
        img_tokens,
        xyz,
        importance,
        lang_tokens,
        num_img,
        h,
        w,
        keep_ratio,
        min_fine_tokens,
        corr_radius,
        corr_topk,
    ):
        # img_tokens: [B, V*H*W, C]
        # xyz: [B, V*H*W, 3]
        # importance: [B, V*H*W]
        bs, total_tokens, dim = img_tokens.shape

        stride = self._coarse_stride(h, w, keep_ratio)
        coarse_h = math.ceil(h / stride)
        coarse_w = math.ceil(w / stride)

        img_map = img_tokens.view(bs, num_img, h, w, dim).permute(0, 1, 4, 2, 3)
        img_map = img_map.reshape(bs * num_img, dim, h, w)
        xyz_map = xyz.view(bs, num_img, h, w, 3).permute(0, 1, 4, 2, 3)
        xyz_map = xyz_map.reshape(bs * num_img, 3, h, w)

        coarse_img_map = F.avg_pool2d(img_map, kernel_size=stride, stride=stride, ceil_mode=True)
        coarse_xyz_map = F.avg_pool2d(xyz_map, kernel_size=stride, stride=stride, ceil_mode=True)

        coarse_tokens = coarse_img_map.view(bs, num_img, dim, coarse_h, coarse_w)
        coarse_tokens = coarse_tokens.permute(0, 1, 3, 4, 2).reshape(bs, num_img * coarse_h * coarse_w, dim)
        coarse_xyz = coarse_xyz_map.view(bs, num_img, 3, coarse_h, coarse_w)
        coarse_xyz = coarse_xyz.permute(0, 1, 3, 4, 2).reshape(bs, num_img * coarse_h * coarse_w, 3)

        coarse_seq = torch.cat((lang_tokens, coarse_tokens), dim=1)
        coarse_seq = self.coarse_attn(coarse_seq) + coarse_seq
        coarse_seq = self.coarse_ffn(coarse_seq) + coarse_seq

        lang_len = lang_tokens.shape[1]
        coarse_lang = coarse_seq[:, :lang_len]
        coarse_ctx = coarse_seq[:, lang_len:]

        fine_k = min(total_tokens, max(min_fine_tokens, int(total_tokens * keep_ratio)))
        fine_idx = importance.topk(fine_k, dim=-1).indices
        fine_tokens = self._gather_tokens(img_tokens, fine_idx)
        fine_xyz = self._gather_tokens(xyz, fine_idx)
        fine_views = fine_idx // (h * w)

        pairwise_mask = self._build_pairwise_mask(fine_xyz, fine_views, corr_radius, corr_topk)
        fine_tokens = self.fine_cross_attn(
            fine_tokens,
            context=fine_tokens,
            attn_mask=pairwise_mask,
        ) + fine_tokens

        context_tokens = torch.cat((coarse_lang, coarse_ctx), dim=1)
        fine_tokens = self.fine_context_attn(
            fine_tokens,
            context=context_tokens,
        ) + fine_tokens
        fine_tokens = self.fine_ffn(fine_tokens) + fine_tokens

        dense_coarse = F.interpolate(
            coarse_img_map,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        dense_coarse = dense_coarse.view(bs, num_img, dim, h, w)
        dense_coarse = dense_coarse.permute(0, 1, 3, 4, 2).reshape(bs, total_tokens, dim)

        updated_tokens = img_tokens + dense_coarse
        updated_tokens = self._scatter_tokens(updated_tokens, fine_idx, fine_tokens)

        aux = {
            "fine_idx": fine_idx,
            "fine_views": fine_views,
            "coarse_xyz": coarse_xyz,
        }
        return updated_tokens, coarse_lang, aux
