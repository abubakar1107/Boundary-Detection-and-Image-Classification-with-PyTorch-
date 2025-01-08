import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Phase2.Code.Network.Network_torch import CIFAR10Model, custom_Resnet, custom_Resnext, custom_DenseNet
from Train_torch import CustomCIFAR10Dataset  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(model, device, test_loader):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0  # Initialize test loss
    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels.long()) 
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).cpu().numpy())


    test_loss /= total  # Calculate the average loss
    accuracy = 100 * correct / total #  Calculate the accuracy

    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    print(f'Average test loss: {test_loss:.4f}')

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_targets, all_predicted)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=True, title='Normalized confusion matrix')
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = CustomCIFAR10Dataset(img_dir='./Phase2/CIFAR10/Test', label_file='./Phase2/Code/TxtFiles/LabelsTest.txt', transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Model instantiation
    """
    Decomment the model you want to use to test on the testset
    """
    model = CIFAR10Model().to(device)
    # model = custom_Resnet().to(device)
    # model = custom_Resnext().to(device)
    # model = custom_DenseNet().to(device)


    # Load your trained model
    """
    Decomment the model you want to load to use to test on the testset
    """
    model.load_state_dict(torch.load('./Phase2/Checkpoints/cifar10_model.pth'))
    # model.load_state_dict(torch.load('./Phase2/Checkpoints/custom_ResNet_model.pth'))
    # model.load_state_dict(torch.load('./Phase2/Checkpoints/custom_ResNeXt_model.pth'))
    # model.load_state_dict(torch.load('./Phase2/Checkpoints/custom_DenseNet_model.pth'))
    
    test(model, device, test_loader)
