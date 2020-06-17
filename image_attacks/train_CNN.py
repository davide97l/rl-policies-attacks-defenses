from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os


class CNN(torch.nn.Module):
  """Basic CNN architecture."""

  def __init__(self, in_channels=1):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
    self.conv2 = nn.Conv2d(64, 128, 6, 2)
    self.conv3 = nn.Conv2d(128, 128, 5, 2)
    self.fc = nn.Linear(128*3*3, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 128*3*3)
    x = self.fc(x)
    return x


def ld_cifar10():
  """Load training and test data."""
  train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, transform=train_transforms, download=True)
  test_dataset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, transform=test_transforms, download=True)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
  return EasyDict(train=train_loader, test=test_loader)


if __name__ == '__main__':

  # Define training parameters
  epochs = 10

  # Load training and test data
  data = ld_cifar10()

  # Instantiate model, loss, and optimizer for training
  net = CNN(in_channels=3)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    net = net.to(device)
  loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

  # Train vanilla model
  total, correct = 0, 0
  net.train()
  for epoch in range(1, epochs + 1):
    train_loss = 0.
    for x, y in data.train:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      logits = net(x)
      loss = loss_fn(logits, y)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      total += y.size(0)
      _, y_pred = logits.max(1)
      correct += y_pred.eq(y).sum().item()
    print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(epoch, epochs, train_loss, correct / total * 100.))

  # Save model
  if not os.path.exists("models"):
    os.makedirs("models")
  torch.save({
    'model': net.state_dict(),
  }, os.path.join("models/CNN.tar"))

  # Load model
  checkpoint = torch.load("models/CNN.tar")
  net.load_state_dict(checkpoint['model'])

  # Evaluate on clean data
  net.eval()
  total, correct = 0, 0
  for x, y in data.test:
    x, y = x.to(device), y.to(device)
    _, y_pred = net(x).max(1)  # model prediction on clean examples
    total += y.size(0)
    correct += y_pred.eq(y).sum().item()
  print('test acc on clean examples: {:.3f}'.format(correct / total * 100.))