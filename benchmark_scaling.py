import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc

def cleanup():
    """清理显存，防止上一次测试影响下一次"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def benchmark_model(model_name, model, res):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    input_size = (3, res, res)
    dummy_input = torch.randn(1, *input_size).to(device)
    
    cleanup()
    
    try:
        # 1. 静态算力统计
        flops_analysis = FlopCountAnalysis(model, dummy_input)
        flops_analysis.unsupported_ops_warnings(False) # 关闭警告，保持输出整洁
        flops = flops_analysis.total() / 1e9
        
        # 2. 动态显存统计
        with torch.no_grad():
            _ = model(dummy_input) # 预热
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_input) # 测试
            
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        return flops, peak_mem
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [!] {model_name} 在 {res}x{res} 下显存溢出 (OOM)")
            return None, None
        raise e

def plot_results(res_list, vit_data, conv_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制 FLOPs 曲线
    ax1.plot(res_list, [d[0] for d in vit_data], 'o-', label='ViT-S/16', color='blue')
    ax1.plot(res_list, [d[0] for d in conv_data], 's-', label='ConvNeXt-T', color='green')
    ax1.set_xlabel('Input Resolution')
    ax1.set_ylabel('GFLOPs')
    ax1.set_title('Computational Complexity (FLOPs)')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    # 绘制 Memory 曲线
    ax2.plot(res_list, [d[1] for d in vit_data], 'o-', label='ViT-S/16', color='blue')
    ax2.plot(res_list, [d[1] for d in conv_data], 's-', label='ConvNeXt-T', color='green')
    ax2.set_xlabel('Input Resolution')
    ax2.set_ylabel('Peak GPU Memory (MB)')
    ax2.set_title('Memory Usage Scaling')
    ax2.grid(True, linestyle='--')
    ax2.legend()

    plt.suptitle('DINOv3: ViT vs ConvNeXt Performance Scaling', fontsize=16)
    plt.tight_layout()
    plt.savefig('scaling_comparison.png')
    print("\n[√] 曲线图已保存为 scaling_comparison.png")
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 定义测试分辨率序列
    resolutions = [224, 448, 518, 672, 896, 1024]
    
    print("正在加载模型...")
    # 注意：确保你有对应的库支持或者从本地加载
    repo_dir = '/home/wayrobo/0_code/dinov3'
    vit_model = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', pretrained=False)
    conv_model = torch.hub.load(repo_dir, 'dinov3_convnext_tiny', source='local', pretrained=False)
    
    vit_stats = []
    conv_stats = []

    for res in resolutions:
        print(f"测试分辨率: {res}x{res} ...")
        
        v_flops, v_mem = benchmark_model("ViT", vit_model, res)
        vit_stats.append((v_flops, v_mem))
        
        c_flops, c_mem = benchmark_model("ConvNeXt", conv_model, res)
        conv_stats.append((c_flops, c_mem))

    # 打印表格输出
    data = []
    for i, res in enumerate(resolutions):
        data.append([res, vit_stats[i][0], conv_stats[i][0], vit_stats[i][1], conv_stats[i][1]])
    
    df = pd.DataFrame(data, columns=['Res', 'ViT FLOPs', 'Conv FLOPs', 'ViT Mem', 'Conv Mem'])
    print("\n对比结果数据表:")
    print(df.to_string(index=False))
    
    # 绘图
    plot_results(resolutions, vit_stats, conv_stats)
