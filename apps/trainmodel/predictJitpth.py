import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
import os  # 导入os库

# 设置numpy打印选项
np.set_printoptions(precision=4, suppress=True)

def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# 创建一个函数来获得相应的变换
def get_transforms():
    return transforms.Compose([
        transforms.CenterCrop(512),  # 中心裁剪
        transforms.Resize(256),      # 调整大小
        transforms.CenterCrop(224),  # 再次中心裁剪
        transforms.ToTensor(),       # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

def predict(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms()
    # 加载预训练模型
    resnet = models.resnet18()
    # 自定义全连接层以适应新的分类任务
    resnet.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=8),
    )
    # 加载训练好的模型权重
    model = torch.jit.load("models/resnet18_leather_scriptmodel.pth", map_location=device)
    model.eval()
    
    with torch.no_grad():
        # 加载图像，转换，应用预处理
        img = Image.open(img_path).convert("RGB")
        img = transform(img)  # 应用相同的变换
        img = img[None, ...].to(device)  # 添加一个批处理维度并转移到正确的设备
        r = model(img)  # 执行前向传播
        # 打印输出
        print(r)
        print(r.shape)
        # 应用Softmax
        softmax_results = softmax(r[0][:8].cpu().numpy())  # 把结果移回CPU来计算Softmax
        print(softmax_results)
        # 获取预测结果
        result_idx = torch.argmax(r).item()
        print(result_idx)

        # 根据模型的输出结果，读取对应文件夹中的唯一文件名
        target_dir = f"apps/original/{result_idx}"
        file_name = os.listdir(target_dir)[0]  # 读取目录下的第一个（也是唯一一个）文件
        print(f"{img_path} 预测结果为: {file_name}")


if __name__ == '__main__':

    predict("ceramic_texture.jpg")
    predict("fabric1_texture.jpg")
    predict("fabric2_texture.jpg")
    predict("leather_texture.jpg")
    predict("metal_texture.jpg")
    predict("stone_texture.jpg")
    predict("stone2_texture.jpg")
    predict("wood_texture.jpg")