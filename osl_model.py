import torch
import torch.nn as nn

class Siamese(nn.Module):
    """
    The number of convolutional
    filters is specified as a multiple of 16 to optimize performance. The network applies a ReLU activation function
    to the output feature maps, optionally followed by maxpooling with a filter size and stride of 2.
    """
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(                                                                                      # sequential model to chain all the layers together
            nn.Conv2d(1, 64, 10),                                                                                       # convolution layer that takes the image as input
            nn.ReLU(inplace=True),                                                                                      # ReLU activation function
            nn.MaxPool2d(2),                                                                                            # MaxPoolinglayer
            nn.Conv2d(64, 128, 7),                                                                                      # convolution layer
            nn.ReLU(),                                                                                                  # ReLU activation function
            nn.MaxPool2d(2),                                                                                            # MaxPoolinglayer
            nn.Conv2d(128, 128, 4),                                                                                     # convolution layer
            nn.ReLU(),                                                                                                  # ReLU activation function
            nn.MaxPool2d(2),                                                                                            # MaxPoolinglayer
            nn.Conv2d(128, 256, 4),                                                                                     # convolution layer
            nn.ReLU(),                                                                                                  # ReLU activation function
        )

        self.linear = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())                                                # the units in the final convolutional layer are flattened into a single vector.
        self.output = nn.Linear(4096, 1)                                                                             #  one more layer computing the induced distance metric between each siamese twin, which is given to a single sigmoidal output unit ?

    def forward_single_twin(self, x):
        x = self.conv(x)                                                                                                # forwarding x through the defined network twin
        x = x.view(x.size()[0], -1)                                                                                     # resizing x
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_single_twin(x1)                                                                          # getting the output for siamese twin 1
        output2 = self.forward_single_twin(x2)                                                                          # getting the output for siamese twin 2
        distance = torch.abs(output1 - output2)                                                                         # computing the induced distance metric between each siamese twin
        output = self.output(distance)                                                                                  # mapping that distance on a single digit between 0 and 1
        return output
