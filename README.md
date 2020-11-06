Kernel self-attention in deep multiple instance learning
================================================

by Dawid Rymarczyk (<rymarczykdawid@gmail.com>), Adriana Borowa, Jacek Tabor and Bartosz Zieliński

Overview
--------

PyTorch implementation of our paper "Kernel self-attention in deep multiple instance learning":


Installation
------------

Installing Pytorch 1.0.1, using pip or conda, should resolve all dependencies.
Tested with Python 3.7.
Tested on both CPU and GPU.


How to Use
----------
`dataloader.py`: Generates training and test set by combining multiple MNIST images to bags. A bag is given a positive label if it contains one or more images with the label specified by the variable target_number.
If run as main, it computes the ratio of positive bags as well as the mean, max and min value for the number per instances in a bag.

`mnist_bags_loader.py`: Added the original data loader we used in the experiments. It can handle any bag length without the dataset becoming unbalanced. It is most probably not the most efficient way to create the bags. Furthermore it is only test for the case that the target number is ‘9’.

`main.py`: Trains a small CNN with the Adam optimization algorithm.
The training takes 20 epoches. Last, the accuracy and loss of the model on the test set is computed.

`model.py`: The model is a modified LeNet-5, see <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.

Acknowledgments
--------------------

Thanks to Maximilias Ilse and Jakub M. Tomczak for sharing their code. 
