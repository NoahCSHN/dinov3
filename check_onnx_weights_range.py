import onnx
import numpy as np
from onnx import numpy_helper

def check_onnx_weights_safety(onnx_path):
    print(f"[*] 正在加载模型: {onnx_path} ...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    print(f"[*] 开始扫描 {len(graph.initializer)} 个权重张量...")
    
    risky_weights = []
    
    for tensor in graph.initializer:
        # 1. 获取权重数据 (FP32)
        weight = numpy_helper.to_array(tensor)
        
        # 跳过非浮点类型的权重 (如 int64 的 shape 或 indices)
        if weight.dtype != np.float32 and weight.dtype != np.float64:
            continue
            
        # 2. 模拟 FP16 转换
        # 这一步非常关键：我们在 numpy 里强制转成 float16，看看会不会变 inf
        weight_fp16 = weight.astype(np.float16)
        
        # 3. 检查溢出 (Overflow)
        if np.isinf(weight_fp16).any():
            max_val = np.max(np.abs(weight))
            print(f"⚠️  [溢出警报] {tensor.name}")
            print(f"    原始最大值: {max_val:.4f} > 65504")
            risky_weights.append(tensor.name)
            
        # 4. 检查严重下溢 (Underflow - 可选，视模型敏感度而定)
        # 这里只检查非零且变得极小的情况
        small_mask = (np.abs(weight) > 0) & (np.abs(weight) < 6e-5)
        if small_mask.any():
            print(f"ℹ️  [精度丢失] {tensor.name} 部分数值将变为 0")

    if not risky_weights:
        print("\n✅ 体检通过！所有权重都在 FP16 安全范围内 ([-65504, 65504])。")
    else:
        print(f"\n❌ 发现 {len(risky_weights)} 个危险权重层，必须处理！")
        return risky_weights

def generate_fp32_protection(onnx_path):
    print(f"[*] 正在分析模型依赖关系: {onnx_path} ...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 1. 找出所有会有下溢风险的权重名称
    risky_tensors = set()
    for tensor in graph.initializer:
        weight = numpy_helper.to_array(tensor)
        if weight.dtype not in [np.float32, np.float64]: continue
        
        # 检查是否会有非零值被截断为 0 (FP16 最小精度约为 6e-5)
        # 我们设定一个阈值，比如 1e-4 以下的我们认为有风险
        # 但我们只关心那些"非稀疏"的下溢，即原本不是0，变成了0
        fp16_min = 6e-5
        underflow_mask = (np.abs(weight) > 0) & (np.abs(weight) < fp16_min)
        
        if underflow_mask.any():
            risky_tensors.add(tensor.name)

    print(f"[*] 发现 {len(risky_tensors)} 个存在下溢风险的权重张量。")
    print("[*] 正在寻找使用这些权重的计算节点...")

    # 2. 找到使用这些权重的节点 (Layer)
    protected_layers = []
    
    # 建立映射：哪些节点使用了这些权重作为输入
    for node in graph.node:
        for input_name in node.input:
            if input_name in risky_tensors:
                # 只有计算密集型或敏感节点需要保护
                # 比如 MatMul, Conv, Gemm, Scale, Mul, Add
                if node.op_type in ["MatMul", "Gemm", "Conv", "Mul", "Add", "Div"]:
                    protected_layers.append(node.name)
    
    # 去重
    protected_layers = sorted(list(set(protected_layers)))
    
    # 3. 生成 trtexec 命令片段
    precision_str = ",".join([f"{name}:fp32" for name in protected_layers])
    
    print("\n" + "="*60)
    print("🛡️  建议的 TensorRT 混合精度保护策略")
    print("="*60)
    print("请在 trtexec 命令中添加以下参数 (保留其他层为 fp16，强制这些层为 fp32)：")
    print("-" * 20)
    print(f'--layerPrecisions="{precision_str}"')
    print(f'--layerOutputTypes="{precision_str}"')
    print("-" * 20)
    print(f"共保护了 {len(protected_layers)} 个计算节点。")

if __name__ == "__main__":
    check_onnx_weights_safety("./pretrained/dinov3_vits16_512.onnx")
    generate_fp32_protection("./pretrained/dinov3_vits16_512.onnx")
