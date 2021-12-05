import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import os


class Omniglot_Train(Dataset):
    """
    We turn our dataset into a class to be in a format acceptable for the DataLoader class.
    The Dataset class is just acting as a wrapper for our training and testing datasets.
    """

    def __init__(self, path, transform=None):
        """
        Initializes the Train Omniglot Dataset.
        :param path: Given path for the dataset.
        :param transform: Given transform strucure.
        """
        super(Omniglot_Train, self).__init__()                                                                          # inherit from generic Dataset class (check above)
        np.random.seed(0)                                                                                               # makes the random numbers predictable
        self.transform = transform                                                                                      # transform of the dataset
        self.data, self.class_number = self.loadToMem(path)

    def loadToMem(self, path):
        """
        Loads the Dataset to the memory. Goes through the entire dataset and pulls the training images from the samples.
        :param path: Given Path for the Dataset.
        :return data: the datastructure, which contains the trainingsamples
        :return index: the index of the specific trainingsample
        """
        print("Loading training data to memory.")
        data = {}                                                                                                       # data structure for saving the values
        turns = [0, 90, 180, 270]                                                                                       # degrees by which the picture is rotated
        index = 0                                                                                                       # key value for the data dictionary

        for degree in turns:                                                                                            # turns the picture by given degree
            for alphabet_path in os.listdir(path):                                                                      # goes through Alphabets
                for character_path in os.listdir(os.path.join(path, alphabet_path)):                                    # goes through characters of the given alphabet
                    data[index] = []                                                                                    # initializes dictonary with the index as key and a list of samples per character as value (lists has 20 values) -> exm. key = 0, value = [pic1, pic2, pic3, ..., pic20]
                    for sample_path in os.listdir(os.path.join(path, alphabet_path, character_path)):                   # goes through samples for given characters
                        file_path = os.path.join(path, alphabet_path, character_path, sample_path)                      # path for specific sample of specific alphabet of specific alphabet
                        data[index].append(Image.open(file_path).rotate(degree).convert('L'))                           # add sample to datastructure, rotate it by the given degree and convert it to the greyscale (only stores greyscales, no color)
                    index = index + 1                                                                                   # increments the index by 1 for every character
        print(25 * "*" + "Finished loading." + 25 * "*")
        return data, index

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        """
        Gets training images with an index.
        :param index: Given index for training image.
        """
        label = None
        images1 = None
        images2 = None
        if index % 2 == 1:                                                                                              # if the image of the index is uneven, get an image from the same class
            label = 1.0
            index1 = random.randint(0, self.class_number - 1)                                                           # takes in a random index
            image1 = random.choice(self.data[index1])                                                                   # takes a random image 1
            image2 = random.choice(self.data[index1])                                                                   # takes a random image 2
        else:                                                                                                           # otherwise get a picture from a different class
            label = 0.0
            index1 = random.randint(0, self.class_number - 1)                                                           # takes a random index from a class
            index2 = random.randint(0, self.class_number - 1)                                                           # takes a random index from a different class
            while index1 == index2:                                                                                     # checks if images are from the same class
                index2 = random.randint(0, self.class_number - 1)                                                       # if they do, change the index
            image1 = random.choice(self.data[index1])                                                                   # takes a random image 1
            image2 = random.choice(self.data[index2])                                                                   # takes a random image 2

        if self.transform:
            image1 = self.transform(image1)                                                                             # transforms image 1
            image2 = self.transform(image2)                                                                             # transforms image 2
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class Omniglot_Test(Dataset):

    def __init__(self, path, transform=None, times=200, way=20):
        """
        Initializes the Train Omniglot Dataset.
        :param path: Given path for the dataset.
        :param transform: Given transform strucure.
        """
        super(Omniglot_Test, self).__init__()                                                                           # inherit from generic Dataset class (check above)
        np.random.seed(1)                                                                                               # makes the random numbers predictable
        self.transform = transform                                                                                      # transform of the dataset
        self.times = times                                                                                              # number of samples to test accuracy
        self.way = way                                                                                                  # number of classes (length of support set)
        self.images1 = None
        self.c1 = None
        self.data, self.class_number = self.loadToMem(path)

    def loadToMem(self, path):
        """
        Loads the Dataset to the memory. Goes through the entire dataset and pulls the training images from the samples.
        :param path: Given Path for the Dataset.
        :return data: the datastructure, which contains the testsamples
        :return index: the index of the specific testsample
        """
        print("Loading test data to memory.")
        data = {}
        index = 0

        for alphabet_path in os.listdir(path):                                                                          # goes through Alphabets
            for character_path in os.listdir(os.path.join(path, alphabet_path)):                                        # goes through characters of the given alphabet
                data[index] = []                                                                                        # initializes dictonary with the index as key and a list of samples per character as value (lists has 20 values) -> exm. key = 0, value = [pic1, pic2, pic3, ..., pic20]
                for sample_path in os.listdir(os.path.join(path, alphabet_path, character_path)):                       # goes through samples for given characters
                    file_path = os.path.join(path, alphabet_path, character_path, sample_path)                          # path for specific sample of specific alphabet of specific alphabet
                    data[index].append(Image.open(file_path).convert('L'))                                              # add sample to datastructure, rotate it by the given degree and convert it to the greyscale (only stores greyscales, no color)
                index = index + 1                                                                                       # increments the index by 1 for every character
        print(25 * "*" + "Finished loading." + 25 * "*")
        return data, index

    def __len__(self):
        return self.times * self.way


    def __getitem__(self, index):
        """
        Gets test images.
        :param index: Index for test image.
        """
        index = index % self.way
        label = None
        if index == 0:                                                                                                  # if the image of the index is 0, get an image from the same class
            self.c1 = random.randint(0, self.class_number - 1)
            self.images1 = random.choice(self.data[self.c1])
            images2 = random.choice(self.data[self.c1])                                                                 # gets image from same class
        else:                                                                                                           # otherwise get an image from a different class
            c2 = random.randint(0, self.class_number - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.class_number - 1)
            images2 = random.choice(self.data[c2])                                                                      # gets image from different class

        if self.transform:
            images1 = self.transform(self.images1)                                                                      # transforms image 1
            images2 = self.transform(images2)                                                                           # transforms image 2
        return images1, images2
