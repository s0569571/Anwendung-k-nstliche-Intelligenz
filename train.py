import torch
import utils.constants as C
from osl_model import Siamese
from dataset import Omniglot_Train, Omniglot_Test
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os

if __name__ == '__main__':

    ### VARIABLES                                                                                                       # declare variables -> more info in "./utils/constants.py"
    cuda = C.CUDA
    gpu = C.GPU
    train_path = C.TRAIN_PATH
    test_path = C.TEST_PATH
    show_at = C.SHOW_AT
    test_at = C.TEST_AT
    max_iteration = C.MAX_ITERATION
    loader_workers = C.LOADER_WORKERS
    batch_size = C.BATCH_SIZE
    way = C.WAY
    times = C.TIMES
    learning_rate = C.LEARNING_RATE


    ### CHECK CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu                                                                            # uses the main gpu for training
    print("use gpu:", gpu, "to train.")
    print(f"Cuda is available: {torch.cuda.is_available()}")                                                            # checks if gpu is used
    print("current device: " + str(torch.cuda.current_device()))
    print("count of devices: " + str(torch.cuda.device_count()))

    ### IMAGE TRANFORMATION WITH COMPOSE
    train_set_transform = transforms.Compose([
        transforms.RandomAffine(15),                                                                                    # random affine transformation of the image keeping center invariant
        transforms.ToTensor()                                                                                           # convert a PIL Image or numpy.ndarray to tensor
    ])

    ### GET DATASET
    train_set = Omniglot_Train(train_path, transform=train_set_transform)                                               # prepares the trainset -> transform is set above
    test_set = Omniglot_Test(test_path, transform=transforms.ToTensor(), times=times, way=way)                          # prepares the testset -> is simply transformed to Tensor -> times and way give the n-shot and n-way for the test process

    ### INITIALIZE LOADER
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=loader_workers)               # prepares the train Dataloader -> train loader is initialized with specified batch_size, no shuffle and a certain number of workers -> the workers
    testLoader = DataLoader(test_set, batch_size=way, shuffle=False, num_workers=loader_workers)                        # prepares the test Dataloader

    ### LOSS FUNCTION
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")                                                        # applies loss algorithm -> this loss combines a Sigmoid layer and the BCELoss in one single class - This version is more numerically stable than using a plain Sigmoid

    ### CREATE MODEL
    model = Siamese()                                                                                                   # creates the model defined in -> osl_model
    if cuda:                                                                                                            # if cuda is active, it will send your model to the current gpu devices
        model.cuda()
    model.train()                                                                                                       # sets the module in training mode.


    ### OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)                                                  # uses the Adam optimizers instead of SGD -> after testing it turned out to have a higher precision
    optimizer.zero_grad()                                                                                               # for every mini-batch during the training phase we typically want to explicitly set the gradients to zero before starting to do backpropragation (i.e., updating the Weights and biases)

    ### IMPORTANT VARIABLES                                                                                             # because PyTorch accumulates the gradients on subsequent backward passes.
    accuracies = deque(maxlen=20)                                                                                       # double ended queue -> so that you can append and delete elements from either side of the list (limit of 20)
    final_accuracy = 0.0
    training_loss = []                                                                                                  # create list for the training loss values
    loss_values = 0                                                                                                     # set loss value to zero
    loss_scores = list()

    ### TRAINING AND TESTING ###
    for batch_nr, (images1, images2, labels) in enumerate(trainLoader, 1):                                              # the trainloader gets traversed and for every batch nr the list of images1, images2 and the labels gets looked at
        ### TRAINING
        if batch_nr > max_iteration:                                                                                    # if max_iteration value is overstepped, end the training
            break
        if cuda:
            images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()                                    # sets to cuda for computing gradients
        images1, images2, labels = Variable(images1), Variable(images2), Variable(labels)                               # computes gradients -> Variable is a wrapper around a PyTorch Tensor and allows us to automatically compute gradients
        optimizer.zero_grad()                                                                                           # sets the gradients of all the optimized torch tensors to zero
        output = model.forward(images1, images2)                                                                        # uses the "forward" method from the "osl_model" to feed the images through the model
        loss = loss_function(output, labels)                                                                            # computes the loss with the defined algorithm above
        loss_values = loss_values + loss.item()                                                                         # adds the loss items to the current loss_values
        loss.backward()                                                                                                 # use the backpropagation function on the loss
        optimizer.step()                                                                                                # updates all the parameters

        if batch_nr % show_at == 0:                                                                                     # show every "show_at"th batch
            if batch_nr % 100 == 10:                                                                                    # print "Training" every 100th batch
                print("\n" + "-" * 14 + "Training" + "-" * 14)
            print(f"Batch Nr.: {batch_nr} --- loss: {round(loss_values/show_at, 5)} --- ")
            loss_values = 0                                                                                             # resets the loss value
        ### TESTING
        if batch_nr % test_at == 0:                                                                                     # tests the dataset every "test_at"th value
            correct, error = 0, 0
            for i, (test_images1, test_images2) in enumerate(testLoader, 1):                                            # takes a test sample pair from the testloader
                if cuda:                                                                                                # checks if cuda applicable
                    test_images1, test_images2 = test_images1.cuda(), test_images2.cuda()                               # sets to cuda for computing gradients
                test_images1, test_images2 = Variable(test_images1), Variable(test_images2)                             # computes gradients
                output = model.forward(test_images1, test_images2).data.cpu().numpy()                                   # uses the "forward" method from the model to feed the images through the model and creates a numpy array
                prediction = np.argmax(output)                                                                          # returns the indices of the maximum values along the output axis (only contains values 0,1) -> so it returns the indice of 1 (which is equal to true)
                if prediction == 0:                                                                                     # if the value 1 is at the 0th position, the prediction was correct
                    correct += 1
                else:                                                                                                   # if the value 0 is at the 0th position, the prediction was false
                    error += 1
            print("\n"+"*" * 31 + "Testing" + "*" * 32)                                                                 # formatting
            print(f"[{batch_nr}]\tTest set\tcorrect:\t {correct}\tfalse:\t{error}\t"
                  f"precision:\t{correct * 1.0 / (correct + error)}")
            print("*" * 70)
            accuracies.append(correct * 1.0 / (correct + error))                                                        # appends the accuracy to the queue
        training_loss.append(loss_values)                                                                               # appends the loss_values

    ### ACCURACY CALCULATION
    for accuracy in accuracies:
        final_accuracy = final_accuracy + accuracy
    final_accuracy = final_accuracy / 20

    print("=" * 70)                                                                                                     # calculates the final accuracy of the last 20 iterations
    print(f"The final accuracy score is: {final_accuracy:.3%}!")                                                        # format the final accuracy to a percentage (exact to 3)
    print("=" * 70)

    ### FINAL LOSS PLOT
    for i, val in enumerate(training_loss):
        if i % show_at == 0:
            loss_scores.append(val)

    plt.plot(range(len(loss_scores)), loss_scores)
    plt.title("Siamese Network Loss Values")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss Value")
    plt.show()
