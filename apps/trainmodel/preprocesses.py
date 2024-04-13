import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def get_initial_transforms():
    """
    为每张图片生成一个新的变换管道，以确保随机性。
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),  # 确保输入是三通道RGB
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 的概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 50% 的概率垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2),  # 随机调整亮度和对比度
        transforms.CenterCrop(768),  # 随机中心裁剪
        transforms.ToTensor()  # 最后将图片转换为张量
    ])

def process_and_save_images(input_dir, output_dir, num_images=32):
    """
    处理原始图片并在指定目录保存多个变体。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.png')]
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)  # 确保每个类别的文件夹存在
        
        for image_path in images:
            image = Image.open(image_path)
            for i in range(num_images):
                transform = get_initial_transforms()  # 为每张图片获取新的变换
                transformed_image = transform(image)
                save_path = os.path.join(class_output_dir, f"image_{i}.png")
                save_image(transformed_image, save_path)  # 使用 save_image 来保存张量为图像文件

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '..', 'original')
    output_dir_train = os.path.join(current_dir, '..', 'datasets', 'leather', 'training_data')
    output_dir_test = os.path.join(current_dir, '..', 'datasets', 'leather', 'testing_data')

    # 处理并保存训练和测试图像
    process_and_save_images(input_dir, output_dir_train, num_images=16)  # 修改为需要的变体数量
    process_and_save_images(input_dir, output_dir_test, num_images=16)
