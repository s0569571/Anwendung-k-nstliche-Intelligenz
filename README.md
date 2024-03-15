# "Siamese Neural Network"



This repository implements a Siamese Neural Network for One-Shot Image Recognition. 
It consists out of 50 different alphabets and 1000 classes with 20 samples per class. The data is not artifical and was written by a broad spectrum of humans.

This picture shows a small overview of the different alphabets and letters.

![oneshot](https://github.com/NicoSchultze/One-Shot-Network/assets/87664933/77707578-6bc4-4d8c-a995-771ddd342858)

Nextly you can see the network structure of the siamese Network. As you can see many convolutional layers are used to achieve the impressive one-shot feat. For more detailled infos see the official paper below.

![structure](https://github.com/NicoSchultze/One-Shot-Network/assets/87664933/0bb4865f-819e-4012-af29-61f01ca07bc8)
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
