import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# 1. 环境准备：加载模型
# 注意：'dinov3_vitl14_ade20k' 表示在 ADE20K 数据集上做过线性拟合的模型
device = "cuda" if torch.cuda.is_available() else "cpu"
local_repo_dir = '/home/wayrobo/0_code/dinov3'
convnext_weights = '/home/wayrobo/0_code/dinov3/dinov3/models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
vits_weights = '/home/wayrobo/0_code/dinov3/dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
# model = torch.hub.load(local_repo_dir, 'dinov3_convnext_tiny', source='local', weights=convnext_weights)
model = torch.hub.load(local_repo_dir, 'dinov3_vits16', source='local', weights=vits_weights)
model.to(device).eval()

# 2. 图像预处理
# DINOv3 内部对输入尺寸有要求，通常建议使用 14 的倍数，518 是常用尺寸
transform = T.Compose([
    T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def run_segmentation_convnext(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features_dict = model.forward_features(input_tensor)
        
        # 1. 提取 Patch Tokens (形状通常为 [1, 1024, 768])
        # 注意：ConvNeXt-Tiny 的通道数通常是 768
        tokens = features_dict["x_norm_patchtokens"] 
        b, n, c = tokens.shape 
        
        # 2. 动态计算空间维度 (Spatial Dimension)
        # ConvNeXt 通常下采样 32 倍。如果输入 512，则特征图为 16x16
        # 如果输入 224，则特征图为 7x7
        h_feat = input_tensor.shape[-2] // 32
        w_feat = input_tensor.shape[-1] // 32
        
        # 3. 变形并调整维度顺序: [B, N, C] -> [B, C, H, W]
        features = tokens.reshape(b, h_feat, w_feat, c).permute(0, 3, 1, 2)
        print(f"Successfully reshaped to: {features.shape}")

        # 4. 后续 PCA 可视化逻辑
        # 展平空间维度进行 PCA: [C, H*W] -> [H*W, C]
        feat_flat = features.reshape(c, -1).t() 
        
        # PCA 降维到 3 维
        feat_centered = feat_flat - feat_flat.mean(dim=0)
        U, S, V = torch.pca_lowrank(feat_centered, q=3)
        pca_feat = torch.matmul(feat_centered, V[:, :3])
        
        # 归一化并恢复形状
        pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
        pca_img = pca_feat.reshape(h_feat, w_feat, 3).cpu().numpy()
        
    # 5. 上采样并显示
    pca_res = Image.fromarray((pca_img * 255).astype(np.uint8))
    # 使用 BILINEAR 缩放让结果更平滑
    pca_res = pca_res.resize(img.size, resample=Image.BILINEAR)
    
    return img, np.array(pca_res)

def run_fusion_segmentation_convnext(img_path):
    img = Image.open(img_path).convert('RGB')
    # 建议使用 512 这种较大的尺寸，以便观察细节
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 获取底层 backbone 实例
    # 根据 DINOv3 封装，通常在 model.backbone
    trunk = model.backbone if hasattr(model, 'backbone') else model

    with torch.no_grad():
        # --- 第一步：手动推进到 Stage 1 (下采样 4 倍) ---
        # 运行第 0 个下采样层 (Stem)
        x = trunk.downsample_layers[0](input_tensor)
        # 运行第 0 个 Stage
        feat_s1 = trunk.stages[0](x) # 形状: [1, 96, H/4, W/4]
        
        # --- 第二步：继续运行到最后得到 Stage 4 ---
        # 这一步我们可以直接调用原有的接口获取最强语义特征
        features_dict = model.forward_features(input_tensor)
        tokens = features_dict["x_norm_patchtokens"]
        
        # 重构 s4 形状: [1, 768, H/32, W/32]
        b, n, c = tokens.shape
        h_s4, w_s4 = input_tensor.shape[-2] // 32, input_tensor.shape[-1] // 32
        feat_s4 = tokens.reshape(b, h_s4, w_s4, c).permute(0, 3, 1, 2)

        # --- 第三步：多尺度融合 (Skip Connection) ---
        # 将语义最强的 s4 (低分辨率) 上采样到 s1 的高清尺寸
        feat_s4_up = F.interpolate(feat_s4, size=feat_s1.shape[-2:], mode='bilinear')
        
        # 拼接特征：[1, 96 + 768, H/4, W/4] = [1, 864, 128, 128]
        fusion_feat = torch.cat([feat_s1, feat_s4_up], dim=1)

        # --- 第四步：PCA 可视化 ---
        # 这里的计算量由于分辨率提升而变大，我们只采样部分像素点计算 PCA 矩阵
        b_f, c_f, h_f, w_f = fusion_feat.shape
        feat_flat = fusion_feat.reshape(c_f, -1).t() # [N, 864]
        
        # 减均值
        feat_centered = feat_flat - feat_flat.mean(dim=0)
        # 快速 PCA
        U, S, V = torch.pca_lowrank(feat_centered, q=3)
        pca_feat = torch.matmul(feat_centered, V[:, :3])
        
        # 归一化并恢复空间形状
        pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
        pca_img = pca_feat.reshape(h_f, w_f, 3).cpu().numpy()

    # 5. 上采样回原图 (从 128x128 缩放比 16x16 锐利得多)
    pca_res = Image.fromarray((pca_img * 255).astype(np.uint8))
    # 注意：此时因为有 Stage 1 约束，用 BILINEAR 也会非常清晰
    pca_res = pca_res.resize(img.size, resample=Image.BILINEAR)
    
    return img, np.array(pca_res)

def run_vit_segmentation(img_path, angle=0):
    img_raw = Image.open(img_path).convert('RGB')
    # transform = get_transform(angle)
    input_tensor = transform(img_raw).unsqueeze(0).to(device)

    with torch.no_grad():
        # 3. 提取 ViT 特征
        # 对于 ViT，我们取最后一层的 Patch Tokens
        # n=1 表示取最后一层中间输出
        intermediate_layers = model.get_intermediate_layers(input_tensor, n=1)
        
        # ViT 输出通常包含 [CLS] + [Patch Tokens]
        # DINOv3 的接口通常直接返回 Patch Tokens 序列 [1, 1369, 384]
        tokens = intermediate_layers[0] 
        b, n, c = tokens.shape
        
        # 4. 空间重构 (Spatial Reshape)
        # 518 / 14 = 37
        w_feat = h_feat = int(n**0.5) 
        features = tokens.reshape(b, h_feat, w_feat, c).permute(0, 3, 1, 2)
        
        # 5. PCA 降维可视化
        feat_flat = features.reshape(c, -1).t()
        feat_centered = feat_flat - feat_flat.mean(dim=0)
        U, S, V = torch.pca_lowrank(feat_centered, q=3)
        pca_feat = torch.matmul(feat_centered, V[:, :3])
        
        # 归一化并恢复形状
        pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
        pca_img = pca_feat.reshape(h_feat, w_feat, 3).cpu().numpy()

    # 6. 后处理：缩放回原图
    pca_res = Image.fromarray((pca_img * 255).astype(np.uint8))
    # ViT 的特征边界非常锋利，使用 BILINEAR 缩放效果很好
    pca_res = pca_res.resize((518, 518), resample=Image.BILINEAR)

    # 关键修改：同时返回原始图像和结果
    # 我们为了方便对比，把原始图像也 resize 到 518
    img_resized = img_raw.resize((518, 518))

    return img_resized, pca_res

# 运行并显示
img, seg_pca = run_vit_segmentation("uLong.jpg")
plt.imshow(seg_pca)
plt.title("DINOv3-ConvNeXt Semantic Feature Map")
plt.show()
