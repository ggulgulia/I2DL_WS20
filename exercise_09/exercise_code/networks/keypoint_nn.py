"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        self.batch_size          = hparams.get('batch_size', 32)
        self.lr                  = hparams.get('lr', 1e-4)
        self.dropout_probability = hparams.get('dropout_probability', 0.5)
        self.activation = hparams.get('activation', nn.ReLU())
        #assuming input image (N, C,H,W) = (N, 1,96,96)
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )

        ### 32 layers of size 48 x 48 after 1st layer ###
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                       )
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                       )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                       )
        self.fc1 = nn.Linear(256*6*6, 416)
        self.fc2 = nn.Linear(416, 30)
        self.dropout = nn.Dropout(0.5)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        N = x.shape[0]
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
