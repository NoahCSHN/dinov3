import torch
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as TVT
import torch.nn.functional as F
import gc

# 1. 在类外部定义官方推荐的 80 个模板（精简版）
PROMPT_TEMPLATES = [
    "a photo of a {}.", "a rendering of a {}.", "a cropped photo of the {}.",
    "the photo of a {}.", "a photo of my {}.", "a photo of the cool {}.",
    "a close-up photo of a {}.", "a bright photo of the {}.",
    "a dark photo of the {}.", "a blurry photo of the {}."
]

class GolfCourseAnalyzer:
    def __init__(self, repo_dir, model_name, backbone_w, dinotxt_w, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = labels
        
        print(f"🏗️  正在通过 Hub 加载 {model_name}...")
        # 1. 使用官方 Hub 入口 (source='local' 避开了网络请求且使用了本地配置)
        self.model, self.tokenizer = torch.hub.load(
            repo_dir, 
            model_name, 
            weights=dinotxt_w, 
            backbone_weights=backbone_w,
            source='local'
        )
        
        # 2. 极致显存优化：转为半精度并切换到评估模式
        self.model = self.model.to(self.device).half().eval()

        # 3. 预处理算子 (DINO 标准)
        self.transform = TVT.Compose([
            TVT.Resize((224, 224), interpolation=TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 5. 预计算文本特征 (Zero-shot 核心)
        self.text_features = self._prepare_text_features()
        print("✅ 分析器初始化完成，准备开始高尔夫球场评估。")

    # 2. 修改类内部的特征准备函数
    @torch.no_grad()
    def _prepare_text_features(self):
        text_feats = []
        for class_name in self.labels:
            # 对每个类别，生成多个提示词
            texts = [template.format(class_name) for template in PROMPT_TEMPLATES]
            tokens = self.tokenizer.tokenize(texts).to(self.device)
        
            with torch.amp.autocast('cuda'):
                # 提取 2048 维特征
                feats = self.model.encode_text(tokens) 
                # 官方切片：取后半部分对齐 Patch
                feats = feats[:, feats.shape[1] // 2 :] 
                # 归一化并取平均值 (Ensemble)
                feats = F.normalize(feats, p=2, dim=-1)
                feats = feats.mean(dim=0) 
                # 再次归一化得到最终类中心
                feats = F.normalize(feats, p=2, dim=-1)
                text_feats.append(feats)
            
        return torch.stack(text_feats) # [num_classes, 1024]

    # 3. 修改推理函数，加入 Logit Scale
    @torch.no_grad()
    def analyze_image(self, img_path):
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).half()

        with torch.amp.autocast('cuda'):
            _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img_tensor)
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        
            # 计算相似度
            similarity = patch_tokens.squeeze(0) @ self.text_features.T
        
            # --- 核心改进：手动增加对比度 (Logit Scaling) ---
            # 如果模型有 logit_scale 就用模型自带的，没有就手动设为 20.0
            scale = getattr(self.model, 'logit_scale', torch.tensor(20.0)).exp().item()
            similarity = similarity * 20.0 # 强制放大差异，让 argmax 更果断
        
            mask = similarity.argmax(dim=-1).reshape(14, 14).cpu().numpy()
        
        return mask, np.array(img_pil)

    def save_result(self, mask, original_img, save_path):
        """将分割掩码叠加到原图并保存"""
        h, w = original_img.shape[:2]
        # 将 14x14 的 mask 放大到原图尺寸
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 定义颜色映射 (根据标签数量定义)
        colors = [
            [0, 255, 0],   # Green Grass -> 绿色
            [0, 255, 255], # Sand Bunker -> 黄色
            [255, 0, 0],   # Water -> 蓝色
            [128, 128, 128] # Others -> 灰色
        ]
        
        overlay = original_img.copy()
        for i, color in enumerate(colors[:len(self.labels)]):
            overlay[mask_resized == i] = overlay[mask_resized == i] * 0.5 + np.array(color) * 0.5
            
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def batch_process(self, input_dir, output_dir, limit=100):
        """批量处理入口"""
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:limit]
        print(f"🚀 开始处理 {len(files)} 张图片...")
        
        for i, filename in enumerate(files):
            img_path = os.path.join(input_dir, filename)
            save_path = os.path.join(output_dir, f"result_{filename}")
            
            mask, original_img = self.analyze_image(img_path)
            self.save_result(mask, original_img, save_path)
            
            if (i + 1) % 10 == 0:
                print(f"📊 进度: {i + 1}/{len(files)} | 显存占用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                torch.cuda.empty_cache()
                gc.collect()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 配置信息
    CONFIG = {
        "repo_dir": "/home/wayrobo/0_code/dinov3",
        "model_name": "dinov3_vitl16_dinotxt_tet1280d20h24l",
        "backbone_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinotxt_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        "input_pics": "/home/wayrobo/0_code/dinov3/dataset/dinov3_test", # 请修改为你的图片路径
        "output_dir": "./golf_results",
        "queries": ["green fairway grass", "sand bunker trap", "water pond", "protective golf safety net", "golf distance yardage marker", "trees and bushes"]
    }

    analyzer = GolfCourseAnalyzer(
        CONFIG["repo_dir"], CONFIG["model_name"], 
        CONFIG["backbone_w"], CONFIG["dinotxt_w"], 
        CONFIG["queries"]
    )
    
    analyzer.batch_process(CONFIG["input_pics"], CONFIG["output_dir"])

