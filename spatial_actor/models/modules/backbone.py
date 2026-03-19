import clip
from clip.model import ModifiedResNet

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

###################### modify clip vit and freeze


class CLIPViTBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.patch_size = model.conv1.kernel_size[0]

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # Handle positional embedding with interpolation if needed
        pos_embed = self.model.positional_embedding.to(x.dtype)
        # pos_embed shape: [num_patches + 1, width]
        if pos_embed.shape[0] != x.shape[1] + 1:
            # Interpolate pos_embed
            # Exclude class token
            cls_pos = pos_embed[0:1, :]
            patch_pos = pos_embed[1:, :]
            
            # Original grid size
            orig_size = int((patch_pos.shape[0]) ** 0.5)
            
            # Reshape to image
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            
            # Interpolate
            patch_pos = F.interpolate(patch_pos, size=(H, W), mode='bicubic', align_corners=False)
            
            # Reshape back
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, C)
            
            # Combine
            pos_embed = torch.cat([cls_pos.unsqueeze(0), patch_pos], dim=1).squeeze(0)

        # Add class token
        cls_token = self.model.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        
        x = x + pos_embed
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Remove CLS token and reshape
        x = x[:, 1:, :] # [B, L, C]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W) # [B, C, H, W]

        # ResNet stages:
        # res1: stride 2
        # res2: stride 4
        # res3: stride 8
        # res4: stride 16
        # res5: stride 32

        # ViT-B/32 output is stride 32.
        # ViT-B/16 output is stride 16.
        
        # We assume input image is standard size, but here we just need to return relative scales if possible or upsample.
        # Since downstream expects res1..res5 with specific strides relative to input, 
        # and we only have one low-res feature map, we upsample it.
        
        # Using the actual stride of the ViT patch.
        # If patch_size is 32, current x is at 1/32 scale (res5).
        # We need res1 (1/2), res2 (1/4), res3 (1/8), res4 (1/16).
        
        out = {}

        if self.patch_size == 32:
            out["res5"] = x
            out["res4"] = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            out["res3"] = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            out["res2"] = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
            out["res1"] = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        elif self.patch_size == 16:
            out["res5"] = F.avg_pool2d(x, kernel_size=2, stride=2) 
            out["res4"] = x
            out["res3"] = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            out["res2"] = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            out["res1"] = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError(f"Patch size {self.patch_size} not supported")
        
        return out


class CLIPResNetBackbone(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)

        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


def load_clip(type='RN50'):
    clip_model, clip_transforms = clip.load(type)
    
    if "ViT" in type:
        backbone = CLIPViTBackbone(clip_model.visual)
        # ViT models in CLIP don't have easy way to get these params like ResNet, 
        # but for SpatialActor compatibility we usually don't need output_dim unless we project text
        # If we need text projection, it's in clip_model.text_projection
        # But here we just return the visual backbone.
        normalize = clip_transforms.transforms[-1]
        return backbone, normalize

    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = CLIPResNetBackbone(layers, output_dim, heads)
    backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    return backbone, normalize


def load_imagenet_res50(pretrained=True):
    backbone = ResNetBackbone(pretrained)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return backbone, normalize
