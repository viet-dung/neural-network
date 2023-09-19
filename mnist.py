import numpy as np 
from tensorflow.keras.datasets import mnist
from layer import *
from utils import *
from network import Network

(train, train_label), (test, test_label) =  mnist.load_data()
train = train.reshape(train.shape[0],28*28,1)/255
test = test.reshape(test.shape[0],28*28,1)/255
train_label = one_hot_coding(train_label)

net = Network(mse,mse_prime)
net.add(FCLayer(28**2,100))
net.add(ActivationLayer(ReLU,ReLU_deriv))
net.add(FCLayer(100,10))
net.add(ActivationLayer(tanh,tanh_prime))
net.fit(train[:1],train_label[:1],10,0.1)

test_label = one_hot_coding(test_label)
net.predict(test,test_label)