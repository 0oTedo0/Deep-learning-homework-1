from Network import Model
import numpy as np
from load_data import Load
import seaborn as sns
import matplotlib.pyplot as plt
from Training import alphas,hiddens,Lambdas

print("Hyper-parameters: alphas= ",alphas,", hiddens= ", hiddens," Lambdas= ",Lambdas,sep='',end='\n')
print("File name is lr-[alpha]-hid-[hidden]-l2-[Lambda]-parameters.npz")
print("For example: lr-0.001-hid-100-l2-0.001-parameters.npz")

path = './data'
data = Load(path)

filename=input("Enter the file name:") # "parameters.npz"

parameters = np.load('./model/' + filename)
alpha=parameters['learning_rate']
decay=parameters['decay']
hidden=parameters['hidden']
Lambda=parameters['Lambda']
print('Learning Rate: ',alpha,' hidden layer: ',hidden,' Lambda: ',Lambda,' Decay rate: ',decay,sep='')
model=Model(784,hidden,10,Lambda)
model.fc1.weight=parameters['w1']
model.fc1.bias=parameters['b1']
model.fc2.weight=parameters['w2']
model.fc2.bias=parameters['b2']

test_acc = list()
for test_sample in data.test:
    x, y = test_sample
    y_hat = model.forward(x)
    test_acc.append(y[y_hat.argmax()])
print("Test Accuracy: ", np.mean(test_acc), sep='', end='\n')

sns.heatmap(model.fc1.weight, cmap='Blues')
plt.tight_layout()
# plt.savefig('w1.jpg')
plt.show()
sns.heatmap(model.fc1.bias, cmap='Blues')
plt.tight_layout()
# plt.savefig('b1.jpg')
plt.show()
sns.heatmap(model.fc2.weight, cmap='Blues')
plt.tight_layout()
# plt.savefig('w2.jpg')
plt.show()
sns.heatmap(model.fc2.bias, cmap='Blues')
plt.tight_layout()
# plt.savefig('b2.jpg')
plt.show()

