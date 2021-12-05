# "Anwendung künstlicher Intelligenz"

### To train the network, the following libraries need to be installed:

torch, 
numpy, 
os, 
random, 
Image, 
torchvision, 
matplotlib, 
collections, 

It is also recommended to have Cuda installed on your system, so that you can train the network more efficiently.

Please import the the folders images_background and images_evaluation from: https://github.com/brendenlake/omniglot

Copy those folders in the data folder of this project.

The screenshare and presentation is found at:
https://drive.google.com/drive/folders/1L4nX5piRm61AGy_QKI7bYmMoYDKnEFAJ?usp=sharing

## Sources

These are the sources I have used to implement the network, dataclasses and to get an overall understanding of the paper.

### Official paper:

https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

### Data:

https://github.com/brendenlake/omniglot

### Dataclasses:

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/docs/stable/data.html
https://blog.paperspace.com/dataloaders-abstractions-pytorch/
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

### Network architecture

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
https://medium.com/analytics-vidhya/a-friendly-introduction-to-siamese-networks-283f31bf38cd
https://github.com/gaungalif/siamese.pytorch
https://github.com/fangpin/siamese-pytorch

### Training class:

https://towardsdatascience.com/siamese-networks-line-by-line-explanation-for-beginners-55b8be1d2fc6
https://github.com/Run542968/Siamese_Network_Pytorch
https://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf
