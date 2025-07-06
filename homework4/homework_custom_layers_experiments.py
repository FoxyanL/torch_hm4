import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Function
from models.custom_layers import *
from utils.training_utils import train
from utils.visualization_utils import plot_accuracies, plot_gradient_flow


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_resnet_experiment(block_instance, name):
    model = MiniResNet(block_instance).to(device)
    print(f"Запуск {name}...")
    start = time.time()
    train_acc, test_acc, times = train(model, trainloader, testloader, epochs=5, lr=1e-3, device=device)
    duration = time.time() - start

    plot_accuracies(train_acc, test_acc, title=f"{name} Accuracy")
    plot_gradient_flow(model, title=f"{name} Gradient Flow")

    print(f"Сводка {name}:")
    print(f"   Test Accuracy: {test_acc[-1]:.4f}")
    print(f"   Avg Time/Epoch: {sum(times)/len(times):.2f} sec")
    print(f"   Total Params: {count_parameters(model)}")


if __name__ == '__main__':
    run_resnet_experiment(BasicResidualBlock(16), "BasicResidualBlock")
    run_resnet_experiment(BottleneckResidualBlock(16), "BottleneckResidualBlock")
    run_resnet_experiment(WideResidualBlock(16), "WideResidualBlock")

    # Кастомные блоки
    run_resnet_experiment(AttentionModule(16), "AttentionModule")
    run_resnet_experiment(CustomPooling(), "CustomPooling")
    run_resnet_experiment(CustomSwishActivation(), "SwishActivation")
