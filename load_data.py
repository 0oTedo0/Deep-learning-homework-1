from mnist import MNIST
import numpy as np


class Load:
    """
    This is use for loading data.
    """
    def __init__(self, path,train_num=60000,test_num=10000,type_num=10):
        mndata = MNIST(path)
        # read train dataset
        train_images, train_labels = mndata.load_training()
        self.train_x=np.array(train_images)/256
        # self.train_images = np.reshape(train_images, (60000, 28, 28))
        self.train_y=np.zeros((type_num,train_num))
        self.train_y[train_labels,range(train_num)]=1
        self.train=list(zip(self.train_x,self.train_y.transpose()))
        # read test dataset
        test_images, test_labels = mndata.load_testing()
        self.test_x=np.array(test_images)/256
        # self.test_images = np.reshape(test_images, (10000, 28, 28))
        self.test_y = np.zeros((type_num, test_num))
        self.test_y[test_labels, range(test_num)] = 1
        self.test = list(zip(self.test_x, self.test_y.transpose()))

