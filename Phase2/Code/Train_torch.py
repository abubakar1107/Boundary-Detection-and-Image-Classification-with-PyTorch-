import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Phase2.Code.Network.Network_torch import CIFAR10Model, custom_Resnet, custom_Resnext, custom_DenseNet
import itertools


"""
Variable to store train Lossesand Accuracies over the epochs
"""
train_losses = []
train_accuracies = []

"""
Call for the customized CIFAR10 dataset
"""
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = np.loadtxt(label_file, dtype=int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{idx+1}.png")  # Assuming files are named 1.png, 2.png, ...
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

"""
Load the data with transformations applied
"""
def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CustomCIFAR10Dataset(img_dir='./Phase2/CIFAR10/Train/', label_file='./Phase2/Code/TxtFiles/LabelsTrain.txt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dataset = CustomCIFAR10Dataset(img_dir='./Phase2/CIFAR10/Test/', label_file='./Phase2/Code/TxtFiles/LabelsTest.txt', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    return train_loader, test_loader

""" 
Function to plot confusion matrix 
"""
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f') if normalize else int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

""" 
Training the model on the Training data set
"""
def train(model, train_loader, epochs=10):
    writer = SummaryWriter('runs/CIFAR10_experiment')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    all_labels = []
    all_preds = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            labels = labels.long()  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)
        
        print(f'Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

   
    print('Finished Training')
    writer.close()

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=True,
                          title='Normalized confusion matrix - Training Data')
    plt.show()


    """
    Decomment the model you want to save after training
    """

    # Save the CIFAR_model
    torch.save(model.state_dict(), './Phase2/Checkpoints/cifar10_model.pth')

    # Save the custom_resnet model
    # torch.save(model.state_dict(), './Phase2/Checkpoints/custom_ResNet_model.pth')

    # Save the custom_resnext model
    # torch.save(model.state_dict(), './Phase2/Checkpoints/custom_ResNeXt_model.pth')

    # Save the custom_densenet model
    # torch.save(model.state_dict(), './Phase2/Checkpoints/custom_DenseNet_model.pth')


def plot_metrics(train_accs, train_losses):
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy vs Epochs')
    plt.subplot(1,2,2)
    plt.plot(epochs, train_losses, 'ro-', label='Training Loss vs Epochs')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    train_loader, _ = load_dataset()  
    
    """
    Decomment the model you want to use for training
    """
    model = CIFAR10Model()
    # model = custom_Resnet()
    # model = custom_Resnext()
    # model = custom_DenseNet()

    print(f"Number of trainable parameters: {count_parameters(model)}")
    train(model, train_loader)
    plot_metrics(train_accuracies,  train_losses)


