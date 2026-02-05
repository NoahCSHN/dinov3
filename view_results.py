import fiftyone as fo

#这是你之前脚本里定义的数据集名称
DATASET_NAME = "dinov3_voc_inference" 

def main():
    # 1. 检查数据集是否存在
    if DATASET_NAME not in fo.list_datasets():
        print(f"❌ 错误：找不到名为 '{DATASET_NAME}' 的数据集。")
        print("可能原因：上次推理脚本在写入数据前就报错退出了，或者还没运行过。")
        print("现有数据集列表:", fo.list_datasets())
        return

    # 2. 加载已有数据集
    print(f"[*] 正在加载数据集: {DATASET_NAME} ...")
    dataset = fo.load_dataset(DATASET_NAME)
    
    print(f"✅ 成功加载！包含 {len(dataset)} 张图片。")
    print("="*50)
    print("🚀 服务已启动，请在 Windows 浏览器访问：http://localhost:5151")
    print("👉 按 Ctrl+C 退出")
    print("="*50)

    # 3. 启动 App (不进行任何推理)
    # 注意：这里保留了 WSL 的配置 (0.0.0.0 和 auto=False)
    session = fo.launch_app(dataset, port=5151, address="0.0.0.0", auto=False)

    # 4. 安全挂起
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭...")
    finally:
        session.close()
        print("✅ 服务已安全关闭。")

if __name__ == "__main__":
    main()
