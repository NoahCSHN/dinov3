import torch
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as TVT
import torch.nn.functional as F
import gc

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

    @torch.no_grad()
    def _prepare_text_features(self):
        tokens = self.tokenizer.tokenize(self.labels).to(self.device)
        # 修复 FutureWarning: 使用新的 autocast 写法
        with torch.amp.autocast('cuda'): 
            features = self.model.encode_text(tokens)
            features = features[:, features.shape[1]//2:]
            features = F.normalize(features, p=2, dim=-1) 
        return features

    @torch.no_grad()
    def analyze_image(self, img_path):
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).half()

        with torch.amp.autocast('cuda'):
            # 1. 使用官方推荐的方法获取图像 patch tokens
            # 返回: cls_tokens, register_tokens, patch_tokens
            _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img_tensor)
            
            # 2. 对 patch 特征进行归一化
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1) # [1, 196, 1024]
            
            # 3. 计算余弦相似度
            # 矩阵乘法：[196, 1024] @ [1024, num_labels]
            similarity = patch_tokens.squeeze(0) @ self.text_features.T
            
            # 4. 生成 14x14 的 Mask
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

