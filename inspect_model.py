import torch
import sys
import os

# 1. 注入路径并导入类
sys.path.append(os.path.expanduser("~/0_code/dinov3"))
from dinov3.models.vision_transformer import DinoVisionTransformer

def inspect_checkpoint(checkpoint_path, model):
    print(f"🔍 正在扫描权重文件: {os.path.basename(checkpoint_path)}")
    
    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = ckpt["model"] if "model" in ckpt else ckpt
    model_state = model.state_dict()

    ckpt_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    # --- 1. 检查多余的键 (权重有，代码没有) ---
    unexpected = sorted(list(ckpt_keys - model_keys))
    # --- 2. 检查缺失的键 (代码有，权重没有) ---
    missing = sorted(list(model_keys - ckpt_keys))
    # --- 3. 检查形状不匹配 ---
    mismatched = []
    for key in ckpt_keys & model_keys:
        if checkpoint_state[key].shape != model_state[key].shape:
            mismatched.append(
                f"{key} | Checkpoint: {list(checkpoint_state[key].shape)} vs Model: {list(model_state[key].shape)}"
            )

    # --- 输出报告 ---
    print("\n" + "="*50)
    print("📊 差异诊断报告")
    print("="*50)

    if unexpected:
        print(f"\n❌ 多余键 (Unexpected): {len(unexpected)} 个")
        print("   提示: 这些参数在代码中不存在。")
        for k in unexpected[:10]: print(f"   - {k}")
        if len(unexpected) > 10: print("   ... 等等")
    else:
        print("\n✅ 没有多余键。")

    if missing:
        print(f"\n⚠️ 缺失键 (Missing): {len(missing)} 个")
        print("   提示: 这些参数将使用随机初始化。")
        for k in missing[:10]: print(f"   - {k}")
    else:
        print("\n✅ 没有缺失键。")

    if mismatched:
        print(f"\n📏 形状不匹配 (Shape Mismatch): {len(mismatched)} 个")
        for m in mismatched: print(f"   - {m}")
    else:
        print("\n✅ 所有共同参数形状对齐。")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 按照 ViT-Large 标准配置尝试实例化
    # 如果报错参数名不对，请根据 vision_transformer.py 的 __init__ 修改参数名
    print("🛠️  正在尝试构建本地模型...")
    test_model = DinoVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # 先不加 init_values 和 num_register_tokens，看看脚本怎么报错
    )

    BACKBONE_W = os.path.expanduser("~/0_code/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    inspect_checkpoint(BACKBONE_W, test_model)
