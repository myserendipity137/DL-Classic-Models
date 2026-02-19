import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 检查合并后的配置
    print(OmegaConf.to_yaml(cfg))

    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:{device}")

    # 2. 实例化数据
    print("Loading Data...")
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data)

    # 3. 实例化模型
    print("Initialize Model...")
    model = hydra.utils.instantiate(cfg.model).to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    if cfg.train.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.learning_rate, momentum=0.9)

    # 5. 训练循环
    epochs = cfg.train.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播 & 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100*correct / total

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total=0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    main()