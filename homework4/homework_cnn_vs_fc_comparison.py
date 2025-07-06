import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import time

from models.fc_models import FullyConnectedNet
from models.cnn_models import SimpleCNN, ResidualCNN, ResidualCNNWithReg
from utils.training_utils import train
from utils.visualization_utils import plot_accuracies, plot_confusion_matrix, plot_gradient_flow
from utils.logger import setup_logger


# MNIST

def load_mnist_data(batch_size=128):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_mnist_models():
    return {
        "FCNet": FullyConnectedNet(input_dim=28 * 28, hidden_dims=[512, 256, 128], output_dim=10),
        "SimpleCNN": SimpleCNN(input_channels=1, num_classes=10),
        "ResidualCNN": ResidualCNN(input_channels=1, num_classes=10)
    }


def compare_models_on_mnist():
    batch_size = 128
    epochs = 10
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("mnist_comparison", "results/mnist_comparison/mnist_comparison.log")

    logger.info("Загрузка MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    models = get_mnist_models()

    results = {}

    for name, model in models.items():
        logger.info(f"Тренировка {name}")
        results[name] = evaluate_model(name, model, train_loader, test_loader, device, epochs, lr, logger, dataset='mnist')
        plot_accuracies(results[name]["train_acc"], results[name]["test_acc"], title=f"{name} on MNIST")

    print("MNIST Results")
    print_results_table(results)


# CIFAR-10

def load_cifar10_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def get_cifar10_models():
    return {
        "FCNet": FullyConnectedNet(input_dim=32 * 32 * 3, hidden_dims=[1024, 512, 256], output_dim=10),
        "ResidualCNN": ResidualCNN(input_channels=3, num_classes=10),
        "ResidualCNN+Reg": ResidualCNNWithReg(input_channels=3, num_classes=10)
    }


def compare_models_on_cifar10():
    batch_size = 128
    epochs = 10
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("cifar10_comparison", "results/cifar10_comparison/cifar10_comparison.log")

    logger.info("Загрузка CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size)
    models = get_cifar10_models()

    results = {}

    for name, model in models.items():
        logger.info(f"Тренировка {name}")
        results[name] = evaluate_model(name, model, train_loader, test_loader, device, epochs, lr, logger, dataset='cifar10')
        plot_accuracies(results[name]["train_acc"], results[name]["test_acc"], title=f"{name} on CIFAR-10")
        plot_confusion_matrix(model, test_loader, device, title=f"{name} Confusion Matrix (CIFAR-10)")
        plot_gradient_flow(model, title=f"{name} Gradient Flow")

    print("CIFAR-10 Results")
    print_results_table(results)


def evaluate_model(name, model, train_loader, test_loader, device, epochs, lr, logger, dataset):
    model = model.to(device)

    input_size = (1, 28 * 28) if dataset == "mnist" and name == "FCNet" else \
                 (1, 1, 28, 28) if dataset == "mnist" else \
                 (1, 3, 32, 32)

    try:
        logger.info(f"Сводка для {name}:")
        summary(model, input_size[1:])
    except Exception as e:
        logger.warning(f"Не удалось отобразить сводку для {name}: {e}")

    start_time = time.time()
    train_acc, test_acc, _ = train(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        X, _ = next(iter(test_loader))
        X = X.to(device)
        inf_start = time.time()
        model(X)
        inference_time = time.time() - inf_start

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "training_time": training_time,
        "inference_time": inference_time,
        "total_params": total_params,
    }


def print_results_table(results):
    print(f"{'Model':<20} | {'Train Acc':<10} | {'Test Acc':<10} | {'Train Time (s)':<14} | {'Infer Time (s)':<14} | {'Params':<10}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<20} | {r['train_acc'][-1]:<10.4f} | {r['test_acc'][-1]:<10.4f} | {r['training_time']:<14.2f} | {r['inference_time']:<14.6f} | {r['total_params']:<10}")

if __name__ == "__main__":
    compare_models_on_mnist()
    compare_models_on_cifar10()
