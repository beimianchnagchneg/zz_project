import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, 128)  # num_classes = 10
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x 应该是 (batch_size, 1)，然后我们在这里扩展维度
        x = x.unsqueeze(1)  # (batch_size, 1) -> (batch_size, seq_length=1)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = self.transformer(x, x)  # (batch_size, seq_length, embedding_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)  # (batch_size, num_classes)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载Fashion-MNIST数据集
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = TransformerModel(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 确保标签是整数类型
        labels = labels.long()  # (batch_size,)

        # 前向传播
        outputs = model(labels)  # 使用标签作为模型的输入
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('训练完成！')
