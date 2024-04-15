
import os
import torch
import torch.nn as nn
import torchvision.models as models

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', '..', 'models')
    resnet = models.resnet18()
    resnet.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=8)
    )
     # 直接加载 TorchScript 模型
    model_path = os.path.join(model_dir, 'resnet18_leather_scriptmodel.pth')
    model = torch.jit.load(model_path, map_location=torch.device('cuda'))
    
    model.eval()

    # 确保输入数据也在 CUDA 设备上
    x = torch.randn(1, 3, 224, 224).to('cuda')

    # 转换为 ONNX
    onnx_path = os.path.join(model_dir, 'resnet18_leather_bestjit.onnx')
    torch.onnx.export(model, x, onnx_path, export_params=True)