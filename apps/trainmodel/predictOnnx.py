import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import os

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

def predict(img_path, model_path):
    transform = get_transforms()

    # 初始化ONNX Runtime
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 加载图像，转换，应用预处理
    img = Image.open(img_path).convert("RGB")
    img = transform(img)  # 应用相同的变换
    img = img.unsqueeze(0).numpy()  # 转换为numpy数组并添加一个批处理维度

    # 执行前向传播
    r = session.run(None, {input_name: img})[0]

    # 应用Softmax
    softmax_results = softmax(r[0])  # 使用softmax处理输出
    print(softmax_results)

    # 获取预测结果
    result_idx = np.argmax(softmax_results)
    print(result_idx)

    # 根据模型的输出结果，读取对应文件夹中的唯一文件名
    target_dir = f"apps/original/{result_idx}"
    file_name = os.listdir(target_dir)[0]  # 读取目录下的第一个（也是唯一一个）文件
    print(f"{img_path} 预测结果为: {file_name}")

if __name__ == '__main__':
    model_path = "models/resnet18_leather_bestjit.onnx"  # ONNX模型的路径
    # 调用predict函数，使用模型对多个图像进行预测
    predict("ceramic_texture.jpg", model_path)
    predict("fabric1_texture.jpg", model_path)
    predict("fabric2_texture.jpg", model_path)
    predict("leather_texture.jpg", model_path)
    predict("metal_texture.jpg", model_path)
    predict("stone_texture.jpg", model_path)
    predict("stone2_texture.jpg", model_path)
    predict("wood_texture.jpg", model_path)
