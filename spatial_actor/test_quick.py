import torch
import time

### CUDA_VISIBLE_DEVICES=0 python /home/et24-huanggf/robot/SpatialActor/spatial_actor/test_quick.py
### Ctrl + C 或者手动查进程并 kill -9 pid

# ====== 配置参数 ======
gpu_id = 2                  # 使用哪张GPU
target_mem_gb = 19           # 想占用的显存（GB）
compute_intensity = 60    # 计算强度（越大GPU利用率越高）

# =====================

device = torch.device(f"cuda:{gpu_id}")

# 获取显存信息
total_mem = torch.cuda.get_device_properties(device).total_memory
target_mem = target_mem_gb * 1024**3

# 计算需要分配多少float32
num_elements = target_mem // 4  # float32 = 4 bytes

print(f"目标占用显存: {target_mem_gb} GB")

# 分配显存（张量）
tensor = torch.empty(num_elements, dtype=torch.float32, device=device)

# 初始化（避免懒分配）
tensor.fill_(1.0)

print("显存占用完成，开始计算负载...")

# 持续计算，让GPU utilization > 0
try:
    while True:
        for _ in range(compute_intensity):
            tensor.mul_(1.0000001) # 轻量计算
        torch.cuda.synchronize()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("停止程序，释放显存")
    del tensor
    torch.cuda.empty_cache()