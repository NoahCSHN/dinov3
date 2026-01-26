import torch
import time
import os
import sys
from PIL import Image
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis
import cv2
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast

# --- 关键：将本地代码库添加到 Python 路径 ---
# 这样你可以直接 import 仓库里的模型类
LIB_PATH = '/home/wayrobo/0_code/dinov3'
sys.path.append(LIB_PATH)

# 根据 DINOv3 官方代码结构 import
# 注意：具体 import 路径需参考你下载的仓库版本
from models import vision_transformer as vits 

def load_full_model_offline(backbone_path, head_path):
    # 加载 Backbone
    model = vits.__dict__['vit_large'](patch_size=16)
    model.load_state_dict(torch.load(backbone_path))
    
    # 加载 Linear Head (通常是一个简单的线性层)
    # 对于 ViT-L，in_features=1024, out_features=类别数
    linear_head = torch.nn.Linear(1024, 1000) 
    linear_head.load_state_dict(torch.load(head_path))
    
    model.cuda().half().eval()
    linear_head.cuda().half().eval()
    return model, linear_head

@torch.no_grad()
def run_offline_eval(image_dir, backbone_weight_path, head_weight_path, batch_size=4):
    device = "cuda"
    model = load_model_offline(backbone_weight_path, head_weight_path)
    
    # 预处理流程
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 获取图片列表
    img_names = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:100]
    
    # 性能统计初始化
    torch.cuda.reset_peak_memory_stats()
    start_total = time.time()
    inference_times = []

    # 批量处理
    for i in range(0, len(img_names), batch_size):
        batch_files = img_names[i : i + batch_size]
        batch_tensors = []
        
        for f in batch_files:
            img = Image.open(os.path.join(image_dir, f)).convert('RGB')
            batch_tensors.append(transform(img))
        
        input_data = torch.stack(batch_tensors).to(device).half()

        # 耗时统计
        torch.cuda.synchronize()
        t0 = time.time()
        
        _ = model(input_data)
        
        torch.cuda.synchronize()
        inference_times.append(time.time() - t0)

    # 结果分析
    avg_batch_time = sum(inference_times) / len(inference_times)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # 算力分析 (仅分析单张图)
    dummy_input = torch.randn(1, 3, 224, 224).to(device).half()
    flops = FlopCountAnalysis(model, dummy_input).total()

    print("\n" + "--- Offline Eval Report ---")
    print(f"Model: DINOv3 ViT-L/16 (Half Precision)")
    print(f"Peak GPU Memory: {peak_mem:.2f} MB")
    print(f"Single Image FLOPs: {flops / 1e9:.2f} G")
    print(f"Avg Latency per Batch ({batch_size}): {avg_batch_time*1000:.2f} ms")
    print(f"Total Time for 100 images: {time.time() - start_total:.2f} s")

@torch.no_grad()
def eval_segmentation_with_classification(model, img_tensor, patch_size=16):
    """
    img_tensor: [1, 3, H, W]
    """
    model.eval()
    h, w = img_tensor.shape[-2:]
    nh, nw = h // patch_size, w // patch_size

    with autocast():
        # 1. 提取特征 (DINOv3 常用 forward_features 或 get_intermediate_layers)
        # 假设返回 dict 包含 'x_norm_patchtokens' [1, N, 1024]
        outputs = model.get_intermediate_layers(img_tensor.cuda().half(), n=1)[0]
        
        # 如果是 ViT-L，特征维度是 1024
        # outputs 形状通常是 [1, nh*nw, 1024]
        patch_features = outputs[:, 1:, :] # 排除 CLS token
        cls_token = outputs[:, 0, :]      # 获取全局分类特征

    # ---- 部分 1: 分割可视化 (PCA) ----
    features_np = patch_features.squeeze(0).cpu().float().numpy() # [N, 1024]
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_np) # 降维到 3 维当作 RGB
    
    # 归一化到 0-255
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min())
    
    segmentation_map = (pca_features.reshape(nh, nw, 3) * 255).astype(np.uint8)
    segmentation_map = cv2.resize(segmentation_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- 部分 2: 块分类逻辑 ----
    # 如果没有分类 Head，通常对特征进行简单的 Argmax 或者 相似度匹配
    # 这里模拟输出每个块的主导“类别 ID”
    patch_scores = torch.mean(patch_features, dim=-1).squeeze(0) # 简化演示：用均值模拟响应
    class_ids = (patch_scores * 10).long().cpu().numpy() # 映射到伪类别空间
    class_map = class_ids.reshape(nh, nw)

    return segmentation_map, class_map

if __name__ == "__main__":
    BACKBONES_WEIGHTS = "/home/wayrobo/0_code/dinov3/dinov3/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    HEAD_WEIGHTS = "/home/wayrobo/0_code/dinov3/dinov3/models/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    IMG_DIR = "/home/wayrobo/0_code/dinov3/dataset/dinov3_test/images/"
    run_offline_eval(IMG_DIR, BACKBONES_WEIGHTS, HEAD_WEIGHTS)

