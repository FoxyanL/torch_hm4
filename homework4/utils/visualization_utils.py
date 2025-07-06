import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn


PLOTS_DIR = "plots"

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_accuracies(train_acc, test_acc, title="Accuracy Curves"):
    ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='Train Accuracy', marker='o')
    plt.plot(test_acc, label='Test Accuracy', marker='s')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(model, dataloader, device, title="Confusion Matrix"):
    ensure_dir(PLOTS_DIR)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()

def plot_gradient_flow(model, title="Gradient Flow"):
    ensure_dir(PLOTS_DIR)
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu())
            max_grads.append(param.grad.abs().max().cpu())

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label='Max gradient')
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label='Mean gradient')
    plt.hlines(0, 0, len(ave_grads), lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0)
    plt.xlabel("Layers")
    plt.ylabel("Gradient value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()

def visualize_gradients(model, input_tensor, save_path=None):
    model.eval()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    loss = F.cross_entropy(output, torch.tensor([1], device=input_tensor.device))
    loss.backward()

    grads = input_tensor.grad[0].cpu().detach().numpy()
    plt.imshow(grads[0], cmap='viridis')
    plt.title("Input Gradient")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_feature_maps(model, input_tensor, save_path=None):
    def hook_fn(module, input, output):
        fmap = output[0].detach().cpu().numpy()
        plt.figure(figsize=(10, 5))
        for i in range(min(6, fmap.shape[0])):
            plt.subplot(1, 6, i + 1)
            plt.imshow(fmap[i], cmap='gray')
            plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(hook_fn)
            break

    model(input_tensor)
    handle.remove()

