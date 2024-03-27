import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# 下载CIFAR-10训练数据
dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# 提取图像和标签
images = [np.array(data[0]) for data in dataset]
labels = [data[1] for data in dataset]

# 转换成NumPy数组
images_np = np.array(images)
labels_np = np.array(labels)

# 保存为.npz文件
np.savez("data/cifar10/train.npz", x=images_np, y=labels_np)
