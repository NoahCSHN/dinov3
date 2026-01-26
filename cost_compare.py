import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
import pandas as pd
from tabulate import tabulate

def get_memory_usage():
    """获取当前 GPU 峰值显存占用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0

def benchmark_model(model_name, model, input_res=(3, 518, 518)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    # 构造输入
    dummy_input = torch.randn(1, *input_res).to(device)
    
    # --- 1. 静态统计 (fvcore) ---
    # 统计参数量
    params = parameter_count(model)[""] / 1e6  # 转化为 M (Million)
    
    # 统计算力 (FLOPs)
    # ignore_missing=True 是因为某些算子 fvcore 可能不认识，但不影响大局
    flops_analysis = FlopCountAnalysis(model, dummy_input)
    flops = flops_analysis.total() / 1e9  # 转化为 GFLOPs
    
    # --- 2. 动态监测 (Memory) ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    with torch.no_grad():
        # 预热一次
        _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 正式推理一次以记录峰值
        _ = model(dummy_input)
        
    peak_mem = get_memory_usage()
    
    return {
        "Model": model_name,
        "Params (M)": f"{params:.2f}",
        "FLOPs (G)": f"{flops:.2f}",
        "Peak Mem (MB)": f"{peak_mem:.2f}",
        "Input Res": f"{input_res[1]}x{input_res[2]}"
    }

# --- 执行对比 ---
if __name__ == "__main__":
    results = []
    
    # 1. 加载 ViT-S/16
    print("Benchmarking ViT-S/16...")
    repo_dir = '/home/wayrobo/0_code/dinov3'
    vit_model = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', pretrained=False)
    results.append(benchmark_model("ViT-S/16", vit_model))
    
    # 2. 加载 ConvNeXt-Tiny
    print("Benchmarking ConvNeXt-Tiny...")
    # 注意：确保你有对应的库支持或者从本地加载
    conv_model = torch.hub.load(repo_dir, 'dinov3_convnext_tiny', source='local', pretrained=False)
    results.append(benchmark_model("ConvNeXt-Tiny", conv_model))

    # --- 结果打印 ---
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("DINOv3 Architecture Comparison Report")
    print("="*50)
    print(tabulate(df, headers='keys', tablefmt='grid'))
