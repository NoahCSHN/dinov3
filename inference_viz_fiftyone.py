import fiftyone as fo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import os
from tqdm import tqdm

# --- 🔧 配置区域 ---
REPO_DIR = '/home/wayrobo/0_code/dinov3' 
#MODEL_PATH = 'pretrained/voc_segmenter_epoch_5.pth'
#IMAGE_DIR = '/home/wayrobo/.cache/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset/versions/1/VOC2012_test/VOC2012_test/JPEGImages' 
#DATASET_NAME = "dinov3_voc_inference"
MODEL_PATH = 'pretrained/rellis_golf_best.pth'
#IMAGE_DIR = '/home/wayrobo/.cache/datasets/RELLIS3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node' 
IMAGE_DIR = 'dataset/dinov3_test' 
DATASET_NAME = "dinov3_test_rellis3d"
BATCH_SIZE_SAVE = 10  # 每跑完 10 张图就存一次盘
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# VOC_CLASSES
#HEADER_CLASSES = [
#    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
#    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
#    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
#    'sofa', 'train', 'tvmonitor'
#]
# GLOF_CLASSES
HEADER_CLASSES = [
    'field', 'road', 'Sand', 'Water', 'Bush', 'Person', 'Obstacle' 
]

# --- 模型定义 (保持不变) ---
class DINOv3Segmenter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', pretrained=False)
        #self.head = nn.Conv2d(768, num_classes, kernel_size=1)
        # 🟢 新的 Head (增加空间感知能力)
        self.head = nn.Sequential(
            # 1. 3x3 卷积，融合周围特征
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 2. 1x1 卷积，输出分类
            nn.Dropout(0.1), # 防止过拟合
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict): features = features.get('x_norm_patchtokens', list(features.values())[-1])
        if features.dim() == 3:
            B, N, C = features.shape
            size = int(N ** 0.5)
            features = features.view(B, size, size, C).permute(0, 3, 1, 2)
        logits = self.head(features)
        output = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

def main():
    # 1. 初始化或加载数据集 (不删除旧的！)
    fo.delete_dataset(DATASET_NAME)
    if DATASET_NAME in fo.list_datasets():
        print(f"[*] 加载已有数据集: {DATASET_NAME}")
        dataset = fo.load_dataset(DATASET_NAME)
    else:
        print(f"[*] 创建新数据集: {DATASET_NAME}")
        dataset = fo.Dataset(DATASET_NAME)
        dataset.persistent = True # 确保持久化

    # 2. 获取已经处理过的图片路径，防止重复
    # 这一步实现了“断点续传”
    print("[*] 正在检查已完成的任务...")
    processed_paths = set(dataset.values("filepath"))
    print(f"[*] 已有 {len(processed_paths)} 张图片被处理过。")

    # 3. 筛选出还需要处理的图片
    all_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    todo_files = [f for f in all_files if f not in processed_paths]
    
    if len(todo_files) == 0:
        print("✅ 所有图片都已推理完成！直接启动 App。")
        launch_ui(dataset)
        return

    print(f"[*] 还有 {len(todo_files)} 张图片等待推理...")

    # 4. 加载模型
    print(f"[*] 加载模型权重...")
    model = DINOv3Segmenter(len(HEADER_CLASSES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("❌ 模型文件不存在")
        return
    model.eval()

    # 5. 推理循环 (带 Batch 保存)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_samples = [] # 临时存放
    
    # 使用 try-finally 确保即使中途退出也能保存最后一批
    try:
        for idx, img_path in enumerate(tqdm(todo_files)):
            # --- 推理逻辑 ---
            img_raw = cv2.imread(img_path)
            if img_raw is None: continue
            h, w = img_raw.shape[:2]
            
            img_pil = transforms.ToPILImage()(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                max_probs, _ = torch.max(probs, dim=1)
                confidence = max_probs.mean().item()

            pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- 创建样本 ---
            sample = fo.Sample(filepath=img_path)
            sample["prediction"] = fo.Segmentation(mask=pred_mask_resized)
            sample["confidence"] = confidence
            batch_samples.append(sample)

            # --- 关键：批量保存 ---
            if len(batch_samples) >= BATCH_SIZE_SAVE:
                dataset.add_samples(batch_samples)
                batch_samples = [] # 清空缓存
                
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断！正在保存当前进度...")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        # 保存剩余的零头
        if len(batch_samples) > 0:
            dataset.add_samples(batch_samples)
            print(f"[*] 保存了最后 {len(batch_samples)} 张图片。")
        
        # 设置类别标签
        dataset.default_mask_targets = {i: label for i, label in enumerate(HEADER_CLASSES)}
        dataset.save()
        print("✅ 数据已同步到磁盘。")

        # 启动 UI
        launch_ui(dataset)

def launch_ui(dataset):
    print("\n" + "="*50)
    print("🚀 服务启动中...")
    print("请在浏览器访问：http://localhost:5151")
    print("="*50)

    # 自动关闭 WSL 浏览器尝试，开放 0.0.0.0
    session = fo.launch_app(dataset, port=5151, address="0.0.0.0", auto=False)
    try:
        session.wait()
    except KeyboardInterrupt:
        session.close()

if __name__ == "__main__":
    main()
