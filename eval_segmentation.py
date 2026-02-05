import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

# --- 配置 ---
NUM_CLASSES = 21 
BATCH_SIZE = 8
REPO_DIR = '/home/wayrobo/0_code/dinov3'
WEIGHTS_PATH = './pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
MODEL_NAME = 'dinov3_convnext_tiny'
DEVICE = "cuda"
DATA_ROOT = './voc_data' # 指向你的 VOCdevkit 父目录
MODEL_PATH = 'pretrained/voc_segmenter_epoch_5.pth' # 你的训练权重

# --- 模型定义 (保持一致) ---
class DINOv3Segmenter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load(REPO_DIR, MODEL_NAME, source='local', pretrained=False)
        # 验证时不需要加载 backbone 预训练权重，因为我们会加载整个模型的 state_dict
        self.head = nn.Conv2d(768, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            features = features.get('x_norm_patchtokens', list(features.values())[-1])
        if features.dim() == 3:
            B, N, C = features.shape
            size = int(N ** 0.5)
            features = features.view(B, size, size, C).permute(0, 3, 1, 2)
        logits = self.head(features)
        output = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

# --- 评价指标计算工具类 ---
class MetricLogger:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, targets):
        # preds: [B, H, W], targets: [B, H, W]
        preds = preds.flatten()
        targets = targets.flatten()
        
        # 忽略 255 (边界)
        valid_indices = targets != 255
        preds = preds[valid_indices]
        targets = targets[valid_indices]
        
        # 计算混淆矩阵
        # bincount 统计 (target * num_classes + pred) 的出现次数
        x = preds + self.num_classes * targets
        bincount = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self):
        # IoU = TP / (TP + FP + FN)
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-10) # 加上 epsilon 防止除零
        miou = np.nanmean(iou)
        
        # Pixel Acc = TP / Total
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        return pixel_acc, miou, iou

# --- 数据预处理 ---
class MaskToTensor:
    def __call__(self, target):
        target = np.array(target)
        target[target == 255] = 255 # 保持 255 不变，我们在 MetricLogger 里忽略它
        return torch.from_numpy(target).long()

def evaluate():
    print(f"[*] 正在加载模型: {MODEL_PATH}")
    model = DINOv3Segmenter(NUM_CLASSES).to(DEVICE)
    
    # 加载训练好的权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return
        
    model.eval()

    # 加载验证集
    print("[*] 准备验证集...")
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        MaskToTensor()
    ])
    
    try:
        val_dataset = datasets.VOCSegmentation(
            root=DATA_ROOT, year='2012', image_set='val', download=False,
            transform=img_transform, target_transform=mask_transform
        )
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    metric_logger = MetricLogger(NUM_CLASSES)
    
    print(f"[*] 开始验证 ({len(val_dataset)} 张图片)...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).cpu().numpy() # 转回 CPU 计算指标
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            metric_logger.update(preds, masks)
            
    # 计算最终指标
    pixel_acc, miou, per_class_iou = metric_logger.compute()
    
    # 打印报告
    #VOC_CLASSES = [
    #    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    #    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
    #    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
    #    'sofa', 'train', 'tvmonitor'
    #]

    print("\n" + "="*40)
    print(f"📊 验证结果报告")
    print("="*40)
    print(f"✅ Global Pixel Accuracy: {pixel_acc*100:.2f}%")
    print(f"🚀 Mean IoU (mIoU)      : {miou*100:.2f}%")
    print("-" * 40)
    print("类别详情 (IoU):")
    for i, iou in enumerate(per_class_iou):
        print(f"  {VOC_CLASSES[i]:<15}: {iou*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate()
