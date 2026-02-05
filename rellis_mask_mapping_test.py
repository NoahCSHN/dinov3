import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- 你的配置 ---
# 找一张真实的 Label 图片路径 (请替换为你硬盘上的实际路径)
# 必须是 _label_id 结尾的文件夹里的图片
TEST_MASK_PATH = "/home/wayrobo/.cache/datasets/RELLIS3D/Rellis_3D_pylon_camera_node_label_id/Rellis-3D/00000/pylon_camera_node_label_id/frame000000-1581624652_750.png" 

def check_mapping():
    if not os.path.exists(TEST_MASK_PATH):
        print(f"❌ 找不到测试图片: {TEST_MASK_PATH}")
        return

    # 1. 读取原始 RELLIS Mask
    print(f"[*] 正在读取: {TEST_MASK_PATH}")
    raw_mask = cv2.imread(TEST_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    
    # 统计原始 ID
    unique_ids = np.unique(raw_mask)
    print(f"🔎 [原始 RELLIS ID 分布]: {unique_ids}")
    print("   (请对照 Rellis 官方文档，确认 4是不是草，5是不是树...)")

    # 2. 建立映射表 (这是你训练脚本里的逻辑)
    mapping = np.ones(256, dtype=np.uint8) * 255 # 默认全部是 255 (忽略)
    
    # --- 你的映射规则 ---
    mapping[2] = 0   # Grass -> Field (0)
    mapping[1] = 0   # Dirt -> Field (0)
    mapping[23] = 1  # Concrete -> Road (1)
    mapping[21] = 1  # Asphalt -> Road (1)
    mapping[33] = 2   # Mud -> Sand (2)
    mapping[31] = 3  # Puddle -> Water (3)
    mapping[6] = 3  # Water -> Water (3)
    mapping[4] = 4   # Tree -> Tree (4)
    mapping[19] = 4  # Bush -> Tree (4)
    mapping[17] = 5  # Person (5)
    mapping[27] = 6   # Barrier -> Obstacle (6)
    mapping[18] = 6  # Fence -> Obstacle (6)
    mapping[5] = 6  # Pole -> Obstacle (6)
    # Sky (7) 保持 255
    
    # 3. 执行映射
    mapped_mask = mapping[raw_mask]
    
    # 统计映射后的 ID
    mapped_ids = np.unique(mapped_mask)
    print(f"➡️ [映射后 Golf ID 分布]: {mapped_ids}")
    print("   (期待看到 0, 4, 255 等，不应该有 10, 23 这种大数)")

    # 4. 可视化对比 (关键步骤！)
    # 为了让人眼能看清，我们需要给 0~6 上色
    # 定义一个简单的调色板 (0=绿, 1=灰, 2=黄, 3=蓝, 4=深绿, 5=红, 6=橙, 255=黑)
    viz_img = np.zeros((mapped_mask.shape[0], mapped_mask.shape[1], 3), dtype=np.uint8)
    
    viz_img[mapped_mask == 0] = [0, 255, 0]    # 0 Grass: Green
    viz_img[mapped_mask == 1] = [128, 128, 128]# 1 Road: Gray
    viz_img[mapped_mask == 2] = [0, 255, 255]  # 2 Sand: Yellow
    viz_img[mapped_mask == 3] = [0, 0, 255]    # 3 Water: Blue
    viz_img[mapped_mask == 4] = [0, 100, 0]    # 4 Tree: Dark Green
    viz_img[mapped_mask == 5] = [255, 0, 0]    # 5 Person: Red
    viz_img[mapped_mask == 6] = [255, 165, 0]  # 6 Obstacle: Orange
    viz_img[mapped_mask == 255] = [0, 0, 0]    # 255 Ignore: Black

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Raw Mask (Dark)")
    plt.imshow(raw_mask, cmap='gray') # 看起来会很黑
    
    plt.subplot(1, 2, 2)
    plt.title("Mapped & Colored Mask (Check This!)")
    plt.imshow(viz_img) # 这里应该是彩色的
    
    plt.show()
    print("✅ 请检查右边的图：草地是绿的吗？树是深绿吗？天空是黑的吗？")

if __name__ == "__main__":
    check_mapping()
