import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
    batch_size_test = 1000
    batch_size_train = 64

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    fig