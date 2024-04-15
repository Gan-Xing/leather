import os
from PIL import Image
from torchvision import transforms

# 定义图像转换函数
def get_transforms():
    return transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 函数用于处理和保存转换后的图像
def process_and_save_images(image_paths, output_folder):
    transform = get_transforms()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = transform(img)  # 应用转换

        # 转换为PIL Image以保存
        img = transforms.ToPILImage()(img)
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_name)
        img.save(output_path, format='JPEG')  # 保存图像
        print(f"Processed and saved {output_path}")

# 主函数，列出要处理的图像路径
if __name__ == '__main__':
    image_paths = [
        "ceramic_texture.jpg",
        "fabric1_texture.jpg",
        "fabric2_texture.jpg",
        "leather_texture.jpg",
        "metal_texture.jpg",
        "stone_texture.jpg",
        "stone2_texture.jpg",
        "wood_texture.jpg"
    ]
    output_folder = "processed_images"  # 定义输出文件夹
    process_and_save_images(image_paths, output_folder)
