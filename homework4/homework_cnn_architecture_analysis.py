import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import logging
from utils.training_utils import train
from utils.visualization_utils import (
    plot_accuracies, visualize_gradients, visualize_feature_maps
)
from utils.logger import setup_logger
from models.cnn_models import (
    CNNMixedKernel, CNNKernel,
    CNN2Layer, CNN4Layer, CNN6Layer, ResNetLike
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# эксперименты
def run_experiment(model_class, name):
    logger = setup_logger(name, f"results/architecture_analysis/{name}.log")
    logger.info(f"Начинаем эксперимент над: {name}")

    model = model_class().to(device)
    train_acc, test_acc, times = train(model, trainloader, testloader, epochs=10, lr=1e-3, device=device)

    plot_accuracies(train_acc, test_acc, title=f"{name} Accuracy")
    logger.info(f"Точность тестовой: {test_acc[-1]:.4f}")
    logger.info(f"Время для эпохи: {sum(times)/len(times):.2f}s")
    return test_acc[-1], sum(times)/len(times)

if __name__ == '__main__':
    results = {}

    # анализ разных ядер
    for k in [3, 5, 7]:
        results[f"kernel_{k}x{k}"] = run_experiment(lambda: CNNKernel(k), f"cnn_kernel_{k}x{k}")
    results["kernel_1x1+3x3"] = run_experiment(CNNMixedKernel, "cnn_kernel_1x1_3x3")

    # Влияние глубины
    results["cnn_2layer"] = run_experiment(CNN2Layer, "cnn_2layer")
    results["cnn_4layer"] = run_experiment(CNN4Layer, "cnn_4layer")
    results["cnn_6layer"] = run_experiment(CNN6Layer, "cnn_6layer")
    results["cnn_resnet"] = run_experiment(ResNetLike, "cnn_resnet")

    # Визуализация градиентов и feature map
    model = ResNetLike().to(device)
    dummy_input = next(iter(trainloader))[0][:1].to(device)
    visualize_gradients(model, dummy_input, save_path="plots/2/gradients_resnet.png")
    visualize_feature_maps(model, dummy_input, save_path="plots/2/feature_maps_resnet.png")

    print("Сводка:")
    for name, (acc, time) in results.items():
        print(f"{name}: Test Acc = {acc:.4f}, Avg Time/Epoch = {time:.2f}s")
