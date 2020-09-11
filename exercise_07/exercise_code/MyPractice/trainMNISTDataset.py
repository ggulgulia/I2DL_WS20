import numpy as np
import torch.optim as optim
import torch.nn as nn
from mnist_dataset_and_loader import MNISTDataSetAndLoader
from twoLayerNetwork import Net
import matplotlib.pyplot as plt
import torchvision, os

def get_mnist_data_dir():

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    mnist_data_dir = os.path.join(project_dir, 'datasets')
    print(project_dir)
    print(mnist_data_dir)
    return mnist_data_dir


def main():
    print("hello from main")

    mnist_data_dir = get_mnist_data_dir()

    mnist = MNISTDataSetAndLoader(mnist_root_dir=mnist_data_dir, batch_size=32)
    mnist_dataset = mnist.create()
    mnist_dataloader = mnist.dataLoader()

    #debug  
    images, labels = iter(mnist_dataloader['train']).next()
    print(f"image tensor shape: {images.shape}")
    print(f"labels: {labels}")
    print(f"labels.shape: {labels.shape}")



if __name__ == '__main__':
    main()
