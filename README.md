# Deep_Learning_TP
                  Just below is the report on my differeent experiments with the neural network. You can find other information on the notebook like picture of the losses.
# Lab 2 - Deep Learning with PyTorch: CIFAR10 object classification 

The goal of this lab work is to design a convolutional neural network with the tool Pytorch. 
This network must perform an image classification between 10 classes.

In order to optimize its performances we can play on different hyper parameters. 
Before trying to find the most performing solution I tried to find the influence of the different hyper parameters.

First I kept the initial structure of the neural network with only one convolution layer defined like this

self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

and one pooling layer defined like this

self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

At this moment I changed the size, the stride and the padding of the kernel of the pooling layer in order to have an output which stays at 18\*32\*32 but I will change it and come back to the initial definition when I will search for the most performing configuration.

So with this configuration I tried to modify successively the learning rate, the batch size and the number of epochs.

# LEARNING RATE

In order to see the influence of the learning rate I kept constant values for the batch size and for the number of epochs. My configuration was batch_size = 32 and n_epochs =10. Then I varied the learning rate between 0.01 and 0.0001. I got these results:

learning_rate = 0.1    => Computation Time 119,74s, Ending Validation Loss 2.3,  10%    Accuracy on test images

learning_rate = 0.001  => Computation Time 162,29s, Ending Validation Loss 1.09, 64,11% Accuracy on test images

learning_rate = 0.0008 => Computation Time 167.93s, Ending Validation Loss 1.07, 64,67% Accuracy on test images

learning_rate = 0.0005 => Computation Time 167.61s, Ending Validation Loss 1.03, 65.39% Accuracy on test images

learning_rate = 0.0004 => Computation Time 166.44s, Ending Validation Loss 1.00, 66.42% Accuracy on test images

learning_rate = 0.0003 => Computation Time 168.44s, Ending Validation Loss 1.06, 64.6 % Accuracy on test images

learning_rate = 0.0001 => Computation Time 168.10s, Ending Validation Loss 1.08, 62.52% Accuracy on test images

We can see that reducing the learning rate increases the computation time. Indeed, the learning rate defines how big are the step in the optimization algorithm (descent of the gradient). Thus, if the step size is smaller it will take longer to converge or it will get stuck in an undesirable local minimum.  However, for a smaller learning rater, we also have a better accuracy. But there is a limit because we see that if the learning rate is under 0.0004 then the accuracy reduces. Thus we have an optimal point around 0.0004. We will take this value for the following tests.


# BATCH SIZE :

Now we keep the learning rate and the number of epochs constant and we modify the batch size.  My configuration was n_epochs = 10 and learning_rate = 0.0004. Then I varied the batch size between 8 and 40 and I got these results:

batch_size = 8   => Computation Time 352.84s, Ending Validation Loss 1.11, 64.71 % accuracy on test images 

batch_size = 16  => Computation Time 226.59s, Ending Validation Loss 1.05, 64.41 % accuracy on test images

batch_size = 25  => Computation Time 185.31s, Ending Validation Loss 1.03, 65.02 % accuracy on test images

batch_size = 32  => Computation Time 166.44s, Ending Validation Loss 1.00, 66.42 % accuracy on test images

batch_size = 40  => Computation Time 155.98s, Ending Validation Loss 1.05, 65.11 % accuracy on test images

We can see that the computation time reduces with the batch size. Indeed, the batch size is the size of the set from the training samples which are propagated through the network. If we allow only small batches then we must do more iterations, so it takes more time. For the performances we can see that an increasing batch size does not necessarily increase performances. There is an optimal value which is around 32 so I kept this initial value for the following.

# NUMBER OF EPOCHS

Finally, I studied the influence of the number of epochs with batch_size = 32 and learning_rate = 0.0004. 

n_epochs = 2  => Computation Time 50.81s,  Ending Validation Loss 1.11, 61.29 % accuracy on test images

n_epochs = 4  => Computation Time 67,68s,  Ending Validation Loss 1.01, 65.07 % accuracy on test images

n_epochs = 7  => Computation Time 118.76s, Ending Validation Loss 1.01, 65.23 % accuracy on test images

n_epochs = 8  => Computation Time 136.33s, Ending Validation Loss 1.04, 63.37 % accuracy on test images

n_epochs = 10 => Computation Time 170.28s, Ending Validation Loss 1.08, 64.58 % accuracy on test images

We can see that increasing the number of epochs increases the computation time. Indeed the number of epochs corresponds to the number of passage of the algorithm on all the training sample. Each passage is decomposed in iteration of size of 32 images (the size of the batch size).
In term of accuracy, there is not an optimal value that can be extracted as for the two other parameters. Indeed, we can see that here the optimal value is around 7. However, this depends on the layer we have. It is the same for the other parameters, but it is even more true for the number of epochs. Indeed, there is a moment at which the validation loss curve start diverging. This is because our network overfit too much. It means that it is really accurate on the training set, but it has difficulties to generalize what it learned to unseen data.

So, we will keep n_epochs = 10 for most of the case but we will have to adapt it depending on the configuration.


# Modification of the structure of our layers:

My first idea was to add some convolution layers. As I saw that adding one, two layers did not change a lot the result I added 3 layers to really see the difference. They were defined like this:

self.conv2=nn.Conv2d(18,64,kernel_size=3,stride=1,padding=1)

self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)

self.conv4=nn.Conv2d(128,18,kernel_size=3,stride=1,padding=1)

So, they don’t modify the size. In order to have a result that has not a too small size I kept the pooling defined like this :

self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

At this moment I still didn’t realise that it was not a the best idea to change the pooling layer but we’ll come back on it after.

I got these results

Accuracy of the network on the 40000 train images: 82.20 %

Accuracy of the network on the 10000 validation images: 73.26 %

Accuracy of the network on the 10000 test images: 72.64 % 

We can see that with 4 layers we increased a lot the performances.
But when I tried to add a 5th layer it was not so useful as I got an accuracy that has fallen to 68.48% on test images.

Then I also tried to modify the size of the kernel from 3 to 1 but it was quite similar.

Accuracy of the network on the 40000 train images: 82.84 %

Accuracy of the network on the 10000 validation images: 72.06 %

Accuracy of the network on the 10000 test images: 71.41 %

Computation time: 231.6 and Validation loss of the last epoch : 0.84.

So finally, I kept my 3*3 kernels as it is also often the popular choice.

# Size of trarining and validation set

Then I tried to see the influence of the size of training and validation samples. I used 3 different configurations and I compared my results:

n_training_samples = 36000 ; n_val_samples = 10000 =>  Computation Time 203.68s, Ending Validation loss = 0.86, 71.20% accuracy on test images

n_training_samples = 42000 ; n_val_samples = 8000  =>  Computation Time 218.15s, Ending Validation loss = 0.81, 72.11% accuracy on test images

n_training_samples = 45000 ; n_val_samples = 5000  =>  Computation Time 221.49s, Ending Validation loss = 0.77, 72.29% accuracy on test images


We can see that reducing the size of the training set increases the ending validation loss. This is quite normal because our training period become less general and so our validation loss curve is more likely to diverge quickly. However for the performances we see that after 40000 (for the training set) there is not a huge difference so we will keep our initial 40000 training set / 10000 validation set configuration.

# Finding the best configuration

Then after having seen the influence of almost all-important parameters I tried to find the most performing configuration. To do so I came back to a pooling layer that concentrate information:

self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0). 

And I used convolutions that create more channels (up to 750 for my best performance). 
For every test I adapted the number of epochs in order to limit the overfitting phenomenon. 
I progressed slowly from 72% of accuracy on the test images to 74% and finally I reached my best configuration that you can find on my notebook. I will sum up here the results:

Accuracy of the network on the 40000 train images: 91.00 %

Accuracy of the network on the 10000 validation images: 76.37 %

Accuracy of the network on the 10000 test images: **75.86%** 

Ending Validation loss = 0.77

Computation time : 270.13s

We could improve these performances with several things. First modifying the first convolution layer to add even more channels for example, but I didn’t know if we had the right to do it. Then we could also use different pooling methods (Average pooling for example). Finally, there are so many configurations to test that we could spend even more time to see  which one is better.

# CONCLUSION 

To sum up I would like to say that this lab work helped me a lot to better understand the role and the influence of every component/hyper-parameter of a convolutional neural network. I founded that there are different trade-offs to do. If you increase too much the number of epochs you will start overfit and so lose accuracy, so you must adapt it. Also, if you pool too much your data, they will get smaller and so you must add channels to compensate but if there are too many channels your computation time will too high. 
And I also learned that in deep learning there are configurations that are known to work well but the best one always depends on the environment and on the objective of our network.
