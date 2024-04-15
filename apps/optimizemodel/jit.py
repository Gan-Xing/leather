import torch
import os
import torch.nn as nn
from torch import quantization
from torchvision import models

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', '..', 'models')

    model = models.resnet18()
    model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=8)
    )
    state_dict = torch.load(os.path.join(model_dir, 'best.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    types_to_quantize = {nn.Conv2d, nn.BatchNorm2d, nn.ReLU}
    quant = quantization.quantize_dynamic(model, types_to_quantize, dtype=torch.qint8)

    scr = torch.jit.script(quant)
    torch.jit.save(scr, os.path.join(model_dir, 'resnet18_leather_scriptmodel.pth'))