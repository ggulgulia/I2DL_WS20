from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision

class Dataset():
    def __init__(self):
        pass
    

    def create(self):
        pass

    def dataLoader(self):
        pass

class MNISTDataSetAndLoader():
    """
    A wrapper around pytorch methods to create 
    MNIST Dataset and the corresponding data loader
    for neural networks training

    Optional parameter to constructors are
    
    mnist_root_dir: location where the MNIST data should be downloaded
    batch_size: batch size for training and testing
    transform: desired transformations to be applied
    """
    def __init__(self, mnist_root_dir='.', batch_size=1,
            transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))])):
    
        self.mnist_root_dir = mnist_root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.mnist_classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        self.fashion_mnist_dataset = None
        self.fashion_mnist_dataloader = None

    def create(self, mode = ['train', 'test'], download=False):
        """
        method that creates the MNIST dataset after downloading it

        input parameters:
        mode: list of modes for dataset
        download: if the dataset has to be donwloaded. Should be true if the data is not already present in the self.mnist_root_dir

        returns:
        fashion_mnist_dataset: a dict containg the dataset for each mode represented by the key

        """
        
        if type(mode) is not list:
            raise TypeError('parameter \'mode\' should be a list of string, e.g [\'train\', \'test\']')
        train = None
        fashion_mnist_dataset = {}
        for m in mode:
            if m is 'train':
                train = True
            else:
                train=False

            cur_datset = \
            torchvision.datasets.FashionMNIST(root=self.mnist_root_dir, train=train, download=download, transform=self.transform)

            fashion_mnist_dataset[m] = cur_datset

        self.fashion_mnist_dataset = fashion_mnist_dataset
        return fashion_mnist_dataset


    def dataLoader(self):
        """
        method to create the dataloader from the dataset created for fashionMNIST dataset
        Pre-requisite is that the 'create' method should be executed before this method could be executed

        input parameters:
            None
        returns:
            fashion_mnist_dataloader: a dict containg the dataloaders for each mode represented as its key
        """
        
        if self.fashion_mnist_dataset is None:
            raise Exception("MNISTDataSet.create method must be run to create an appropritae dataloader")
        fashion_mnist_dataloader = {}
        
        dataset_dict = self.fashion_mnist_dataset
        for mode in dataset_dict.keys():
            curr_data_loader = DataLoader(dataset_dict[mode], self.batch_size)
            fashion_mnist_dataloader[mode] = curr_data_loader

        self.fashion_mnist_dataloader = fashion_mnist_dataloader

        return fashion_mnist_dataloader

    def get_class_names(self):
        return self.mnist_classes

####################################################################
#               END OF MNISTDatasetAndLoder class                  #
####################################################################

