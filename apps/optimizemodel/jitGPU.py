import torch
import os
import torch.nn as nn
from torch import quantization
from torchvision import models

# 确保这个脚本作为主程序运行
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', '..', 'models')

    # 加载ResNet18模型
    model = models.resnet18()
    
    # 修改全连接层以适应皮革分类任务
    model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=8)
    )
    
    # 加载预训练的模型权重，确保模型权重与你的任务相适应
    # 使用torch.load加载模型，指定使用GPU进行计算
    state_dict = torch.load(os.path.join(model_dir, 'best.pth'),map_location=torch.device('cuda'))
    
    # 将模型权重加载到模型中
    model.load_state_dict(state_dict)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 将模型转移到GPU上
    model.to('cuda')

    # 定义想要动态量化的层类型，这里选择了卷积层，批标准化层和ReLU激活函数
    types_to_quantize = {nn.Conv2d, nn.BatchNorm2d, nn.ReLU}
    
    # 对指定的层类型进行动态量化，使用qint8类型以减少模型大小和提高推理速度
    # 注意：动态量化通常用于推理阶段
    quant = quantization.quantize_dynamic(model, types_to_quantize, dtype=torch.qint8)
    
    # 将量化后的模型转换为TorchScript，以便于更高效地部署
    scr = torch.jit.script(quant)
    
    # 保存TorchScript模型，以便后续使用
    torch.jit.save(scr, os.path.join(model_dir, 'resnet18_leather_scriptmodel.pth'))
