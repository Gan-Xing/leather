import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
from PIL import Image

class DataSet:
    def __init__(self, root_dir, batch_size, shuffle, num_workers, istrainning):
        super(DataSet, self).__init__()
        self.istrainning = istrainning
        self.dataset = datasets.ImageFolder(root=root_dir,
                                            transform=self.get_transforms())
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_transforms(self):
        if self.istrainning:
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('L')),  # 转换为灰度图
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.RandomAffine(degrees=(-30,30), shear=(-30, 30)),  # 随机倾斜和旋转
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # 随机透视变换
                transforms.RandomApply([  # 随机应用以下变换列表中的变换
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))  # 提高sigma上限以增加模糊深度
                ], p=0.5),  # 50%的概率应用高斯模糊
                transforms.Lambda(lambda x: self.conditional_enhance(x)),
                transforms.CenterCrop(512),  # 随机中心裁剪
                transforms.Resize(256),
                transforms.CenterCrop(224),  # 随机中心裁剪,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('L')),  # 转换为灰度图
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.Lambda(lambda x: self.conditional_enhance(x)),
                transforms.CenterCrop(512),  # 随机中心裁剪,
                transforms.Resize(256),
                transforms.CenterCrop(224),  # 随机中心裁剪,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def estimate_brightness(self, image):
        # 将图像转换为灰度以估计亮度
        grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        return np.mean(grayscale)

    def conditional_enhance(self, image):
        brightness = self.estimate_brightness(image)
        if brightness < 100:  # 设置亮度阈值，根据需要调整
            image = self.enhance_image(image)
        return image

    def enhance_image(self, image):
        np_image = np.array(image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if np_image.ndim == 3:
            for i in range(3):
                np_image[:, :, i] = clahe.apply(np_image[:, :, i])
        else:
            np_image = clahe.apply(np_image)

        gamma = 1.0 / 1.5
        np_image = np.power(np_image / 255.0, gamma) * 255.0

        image = Image.fromarray(np.uint8(np_image))
        return image

    def __len__(self):
        return len(self.dataset.imgs)

    def __iter__(self):
        for data in self.loader:
            yield data


if __name__ == '__main__':
    batch_size = 8
    num_workers = 0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'datasets', 'leather')

    train_dataset = DataSet(os.path.join(output_dir, 'training_data'), batch_size, True, num_workers, True)
    test_dataset = DataSet(os.path.join(output_dir, 'testing_data'), batch_size, False, num_workers, False)

    # train_dataset = DataSet('../datasets/leather/training_data', batch_size, True, num_workers, True)
    # test_dataset = DataSet('../datasets/leather/testing_data', batch_size, False, num_workers, False)
    # print(len(train_dataset))

    # for inputs, labels in train_dataset:
    #     print(inputs.shape)  # 打印图像张量的形状
    #     print(labels.shape)  # 打印标签张量的形状
    #     print(labels[0].item())  # 打印第一个标签的值

