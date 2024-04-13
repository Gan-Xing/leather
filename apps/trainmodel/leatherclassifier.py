# import os
# # 设置环境变量
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
print('torch.cuda.is_available()',torch.cuda.is_available())
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision
from torch import optim
from torchvision import models
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights, GoogLeNet_Weights, ResNet18_Weights

import numpy as np
import os
import matplotlib.pyplot as plt

from dataset import DataSet
from metrics import AccuracyScore

torch.set_printoptions(precision=2, sci_mode=False)


class LeatherClassifier:
    def __init__(self, model, train_data_dir, test_data_dir):
        self.batch_size = 128
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.fc = nn.Sequential(
            # nn.Dropout(0.5),  # Adding Dropout
            # nn.Linear(2048, 13), # Resnet50
            nn.Linear(in_features=512, out_features=5), # Resnet18
        )

        # VGG16
        # self.model.classifier[6] = nn.Sequential(
        #     nn.Dropout(0.5),   # 保留Dropout
        #     nn.Linear(4096, 13) # 修改最后一个全连接层的输出大小
        # )
        self.best_acc = 0
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.total_epoch = 5
        self.lr = 0.01
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = AccuracyScore()
        self.opt = optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad is True],
            lr=self.lr,
            # weight_decay=1e-4  # Adding L2 regularization
        )
        self.print_interval = 2
        self.model_dir = 'models'

        print('self.device',self.device)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            names = os.listdir(self.model_dir)
            if len(names) > 0:
                names.sort()
                name = names[-1]
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    torch.load(os.path.join(self.model_dir, name)))
        self.model = self.model.to(self.device)  # 注意这一行要放在后面

    def save_model(self, epoch):
        # 模型保存
        if epoch == self.total_epoch:
            model_path = os.path.join(self.model_dir, "last.pth")
        else:
            model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_best_model(self, acc):
        if self.best_acc <= acc:  # 等于的时候也更新
            self.best_acc = acc
            model_path = os.path.join(self.model_dir, "best.pth")
            torch.save(self.model.state_dict(), model_path)

    def unnormalize(self, image):
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        return image


    def save_misclassified_images(self, images, epoch, batch):
        folder_path = os.path.join('misclassified_images', f'epoch_{epoch}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path = os.path.join(folder_path, f'misclassified_batch{batch}.png')
        torchvision.utils.save_image(images, image_path)

    def train(self):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        # 1. 加载数据
        trainset = DataSet(root_dir=self.train_data_dir,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              istrainning=True)
        testset = DataSet(root_dir=self.test_data_dir,
                             batch_size=self.batch_size,
                             shuffle=False,
                             num_workers=self.num_workers,
                             istrainning=False)

        for epoch in range(self.total_epoch):
            self.model.train(True)  # Sets the module in training mode.
            train_loss = []
            train_acc = []
            batch = 0
            for inputs, labels in trainset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                acc = self.acc_fn(output, labels)

                train_loss.append(loss.item())
                train_acc.append(acc)
                # if batch % self.print_interval == 0:
                print(f'{epoch + 1}/{self.total_epoch} {batch} train_loss={loss.item()} -- acc={acc.item():.4f}')
                batch += 1

            train_losses.append(np.mean(train_loss))
            train_accuracies.append(np.mean(train_acc))
            print(f'{epoch + 1}/{self.total_epoch} train mean loss {train_loss} -- acc={(train_acc)}')
            print(f'{epoch + 1}/{self.total_epoch} train_losses {train_losses} -- train_accuracies={(train_accuracies)}')
            test_loss = []
            test_acc = []
            batch = 0
            for inputs, labels in testset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)
                acc = self.acc_fn(output, labels)

                test_loss.append(loss.item())
                test_acc.append(acc.item())
                # if batch % self.print_interval == 0:
                print(f'{epoch + 1}/{self.total_epoch} {batch} test_loss={loss.item()} --acc={acc.item():.4f}')
                misclassified_indices = (output.max(1)[1] != labels).nonzero(as_tuple=True)[0]

                # 在模型训练或测试的适当部分添加以下代码
                if misclassified_indices.numel() > 0:
                    # print("Misclassified indices:", misclassified_indices.tolist())
                    misclassified_images = inputs[misclassified_indices]
                    misclassified_images = torch.stack([self.unnormalize(img) for img in misclassified_images])  # 反归一化所有误分类图像
                    self.save_misclassified_images(misclassified_images, epoch, batch)

                batch += 1
            test_losses.append(np.mean(test_loss))
            test_accuracies.append(np.mean(test_acc))
            print(f'{epoch + 1}/{self.total_epoch} test mean loss {test_loss} -- acc={(test_acc)}')
            print(f'{epoch + 1}/{self.total_epoch} test_losses {test_losses} -- test_accuracies={(test_accuracies)}')
            self.save_model(epoch)
            self.save_best_model(acc.item())

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        print(f'{epoch} train mean loss {np.mean(train_loss):.4f} test mean loss {np.mean(test_loss):.4f}'
              f' train mean acc {np.mean(train_acc):.4f} test mean acc {np.mean(test_acc):.4f}')
        self.save_model(self.total_epoch)


if __name__ == '__main__':
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'datasets', 'leather')

    train_data_dir = os.path.join(output_dir, 'training_data')
    test_data_dir = os.path.join(output_dir, 'testing_data')

    model = LeatherClassifier(resnet, train_data_dir, test_data_dir)
    model.train()
