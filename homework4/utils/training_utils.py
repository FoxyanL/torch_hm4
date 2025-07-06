import torch
import time
import logging

def train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_acc, test_acc, times = [], [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        start_time = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        train_accuracy = correct / total
        train_acc.append(train_accuracy)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
        test_accuracy = correct / total
        test_acc.append(test_accuracy)

        epoch_time = time.time() - start_time
        times.append(epoch_time)

        logging.info(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | Time: {epoch_time:.2f}s")

    return train_acc, test_acc, times