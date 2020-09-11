import numpy as np
import torch.optim as optim
import torch.nn as nn, torch
import torchvision.transforms as transforms
from mnist_dataset_and_loader import MNISTDataSetAndLoader
from twoLayerNetwork import Net
import matplotlib.pyplot as plt
import torchvision, os

def get_mnist_data_dir():

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    mnist_data_dir = os.path.join(project_dir, 'datasets')
    print(project_dir)
    print(mnist_data_dir)
    return mnist_data_dir


def get_torch_device():
    """
    helper method to get default device for 
    torch tensors
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def imshow(img):

    """
    helper function to plot 
    an image from a torch.Tensor
    """
    img = img/2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show(block=False)

    return

def main_train():
    """
    use the two layer network created in the file in 
    this directory to learn and identify MNIST Dataset
    """
    device = get_torch_device()

    mnist_data_dir = get_mnist_data_dir()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    batch_size = 32

    mnist = MNISTDataSetAndLoader(mnist_root_dir=mnist_data_dir, transform=transform, batch_size=batch_size)

    mnist_dataset = mnist.create()
    mnist_dataloader = mnist.dataLoader()
    classes = mnist.get_class_names()

    #debug  
    #images, labels = iter(mnist_dataloader['train']).next()
    #print(f"image tensor shape: {images.shape}")
    #print(f"labels: {labels}")
    #print(f"labels.shape: {labels.shape}")
    #imshow(torchvision.utils.make_grid(images))
    #print(net)

    #declare the network
    net = Net(hidden_size=400, activation=nn.ReLU())
    
    loss_fun = nn.CrossEntropyLoss()

    ## notice optimizer takes the network parameters and this is how
    ## optimizer like SGD, Momentum, Adam, etc sees the network weights
    ## sees the network parameters that are optimized
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.1)
    num_epochs = 2
    train_loss_history, train_acc_history = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct      = 0.0
        total        = 0
        for i, data in enumerate(mnist_dataloader['train']):
            X,y = data
            #print(f"X.shape(image tensor): {X.shape}, labels: {y}" )
            X,y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_out = net(X) #forward + backward encapsulated
            #print(f"y_out : {y_out}")

            # y_out has a method grad_fun which refers back to
            # the model parameters and hence calling loss.backwards()
            # back propagates the loss to the network
            # remember y_out is the result of calling net(X), hence y_out has
            # some sort of connection to the network that gave birth to it ;)
            loss = loss_fun(y_out, y)
            loss.backward() # here loss
            optimizer.step() # remember optimizer refers to the net.parameters() when it was first created
            running_loss += loss.item()
            #print(f"loss {loss}\nloss.item(): {loss.item()}")
            _, preds = torch.max(y_out, 1)
            #print(f"preds.shape: {preds.shape}\n preds: {preds}")
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            
            
            #print stats
            if i%500 == 0: #print every 500 mini batches
                running_loss /= 500
                correct /= total
                print("[Epoch: %d Iteration: %5d] loss: %.3f acc: %.2f %%"%(epoch+1, i+1, running_loss, 100*correct))
                train_acc_history.append(correct)
                train_loss_history.append(running_loss)
                running_loss = 0.0
                correct = 0.0
                total = 0.0

    print('FINISH TRAINING')

    #obtain one batch of test images
    dataiter = iter(mnist_dataloader['test'])
    #above dataiter is a generator. 
    #use __next__() on it to obtain the first minbatch
    images, labels = dataiter.__next__()
    images, labels = images.to(device), labels.to(device)

    #predict the output
    y_pred = net(images)
    #convert output to probabilities
    _, predicted = torch.max(y_pred, 1)
    #prepare images to display
    images = images.cpu().numpy()

    #plot the images along with predicted and true label
    fig = plt.figure(figsize=(25,4))
    plt.rcParams["font.size"] = "5"
    for idx in range(32):
        ax = fig.add_subplot(2, 32/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(f"{classes[predicted[idx]]} ({classes[labels[idx]]})",
                color="green" if predicted[idx]==labels[idx] else "red")

    plt.show(block=False)
    return train_loss_history, train_acc_history


def main_plot(train_loss_history, train_acc_history):
    """
    helper functionn to plot training loss and accuracy
    history
    """

    plt.figure()
    plt.plot(train_acc_history)
    plt.plot(train_loss_history)
    plt.title("FashionMNIST")
    plt.xlabel('iteration')
    plt.ylabel('acc/loss')
    plt.legend(['acc', 'loss'])
    plt.show(block=False)
if __name__ == '__main__':
    train_loss_history, train_acc_history = main_train()
    main_plot(train_loss_history, train_acc_history)
