import os
from PIL import Image

# 定义图像转换函数
def get_transforms():
    return Image.open(image_path).crop((left, top, right, bottom))

# 函数用于处理和保存转换后的图像
def process_and_save_images(image_paths, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size  # 获取图像的原始尺寸

        # 计算裁剪框
        new_width = 512
        new_height = 512
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        
        img = img.crop((left, top, right, bottom))  # 裁剪图像

        # 直接保存处理后的图像
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_name)
        img.save(output_path, format='PNG')  # 使用PNG格式保存图像
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
