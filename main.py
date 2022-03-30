import numpy as np
import matplotlib.pyplot as plt
from Network import Model
from random import shuffle
from load_data import Load

np.random.seed(123)
path = './data'
data = Load(path)

alpha = 0.001  # learning rate
decay = 0.98
hidden = 100
Lambda = 1e-3  # regularization coefficient

model = Model(784, hidden, 10, Lambda=Lambda)

epoch = 50
Train_loss_record = list()
Test_loss_record = list()
Train_acc_record = list()
Test_acc_record = list()
# start training
for i in range(epoch):
    print("Epoch ", i, ":", sep='', end=' ')
    curr_alpha=alpha*np.power(decay,i)
    train_loss = 0
    train_acc = list()
    shuffle(data.train)  # randomize samples
    for train_sample in data.train:
        x, y = train_sample
        y_hat = model.forward(x)  # compute estimated label
        train_loss += model.compute_loss(y, y_hat)  # compute loss
        train_acc.append(y[y_hat.argmax()])
        model.BP(y, y_hat, curr_alpha)  # Back propagation
    Train_loss_record.append(train_loss)
    Train_acc_record.append(np.mean(train_acc))
    print("Train Loss: ", train_loss, sep='', end='\t')
    print("Train Acc: ", np.mean(train_acc), sep='', end='\n')

    test_loss = 0
    test_acc = list()
    for test_sample in data.test:
        x, y = test_sample
        y_hat = model.forward(x)  # compute estimated label
        test_loss += model.compute_loss(y, y_hat)  # compute loss
        test_acc.append(y[y_hat.argmax()])
    Test_loss_record.append(test_loss)
    Test_acc_record.append(np.mean(test_acc))
    print("Test Loss: ", test_loss, sep='', end='\t')
    print("Test Acc: ", np.mean(test_acc), sep='', end='\n')

# saving model
name=''.join(['lr-',str(alpha),'-hid-',str(hidden),'-l2-',str(Lambda)])
np.savez(name+'-parameters.npz', learning_rate=alpha, hidden=hidden, Lambda=Lambda,decay=decay,
         w1=model.fc1.weight, b1=model.fc1.bias,
         w2=model.fc2.weight, b2=model.fc2.bias)


# loss & accuracy plot
plt.plot(range(epoch), np.array(Train_loss_record)/60000, 'r--', label='train')
plt.plot(range(epoch), np.array(Test_loss_record)/10000, 'g--', label='test')
plt.legend(fontsize=10)
plt.title("Loss-Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.tight_layout()
# plt.savefig('loss.jpg')
plt.show()
plt.plot(range(epoch), Train_acc_record, 'r--', label='train')
plt.plot(range(epoch), Test_acc_record, 'g--', label='test')
plt.legend(fontsize=10)
plt.title("Accuracy-Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.tight_layout()
# plt.savefig('accuracy.jpg')
plt.show()
