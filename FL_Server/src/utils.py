import json
import torch
from concurrent.futures import as_completed
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset


def write_file(file_name):
    data = {'file_name': file_name}

    files = {
        'json': ('json_data', json.dumps(data), 'application/json'),
        'model': (file_name, open(file_name, "rb"), 'application/octet-stream')}

    return files


def wait_finish(jobs):
    for _ in as_completed(jobs):
        pass
    return "Finish"


def write_text_file(acc, loss, train_loss):
    with open("Result_mnist_128.text", 'w') as f:
        f.write(" ".join(str(item) for item in train_loss))
        f.write("\n")

        f.write(" ".join(str(item) for item in acc))
        f.write("\n")
        f.write(" ".join(str(item) for item in loss))
        f.write("\n")


def get_dataset(dataset):
    data_dir = './data/' + dataset + "/"

    if dataset == "mnist":
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    else:
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
    if dataset == "cifar":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
    return train_dataset, test_dataset


def test_inference(global_model, test_dataset):
    global_model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = global_model(images)

            loss += criterion(outputs, labels).item()
            pred_labels = outputs.argmax(dim=1, keepdim=True)
            correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return accuracy, loss
