def create_training_set():
    class1=[]
    class2=[]
    label = [1,1,0,0]
    net = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for i in range(24):
        class1.append([round(np.random.uniform(low=-10,high=-5)),np.random.uniform(low=0.1,high=.99)])
        class2.append([np.random.uniform(low=0,high=.9),round(np.random.uniform(low=-10,high=-5))])
    for i in range(24):
        class1.append([round(np.random.uniform(low=5,high=10)),np.random.uniform(low=0.1,high=.99)])
        class2.append([np.random.uniform(low=0,high=.9),round(np.random.uniform(low=5,high=10))])
    net.extend(class1)
    net.extend(class2)
    for i in net[4:]:
        if abs(i[0])>1 and abs(i[1])<1:
            label.append(1)
        else:
            label.append(0)
    return np.array(net),np.array(label)
import numpy as np
import random
import matplotlib.pyplot as plt

beta = 1
def sigmoid(x):
    return 1.0/(1.0 + (np.exp(-beta*x)))

def sigmoid_prime(x):
    return beta*sigmoid(x)*(1.0-sigmoid(x))


def CrossEntropy(yHat, y):
    data =  (1-y)/(1-yHat)-(y/yHat)
    if(abs(data)<0.1):
        return 0
    else:
        return data
    
class NeuralNetwork:
    def generate_random_weight(self,layers):
        for i in range(1, len(layers) - 1):
            r = np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        r = np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
    
    def __init__(self, layers):
        
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime
        self.weights = []
        self.weight = self.generate_random_weight(layers)
    
    def calculat_layerWeights(self,h_layer,a):
        for l in range(len(self.weights)):
            dot_value = np.dot(a[l], self.weights[l])
            h_layer.append(dot_value)
            activation = self.activation(dot_value)
            a.append(activation)
        return h_layer,a

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        error_log = []
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            h_layer = [X[i]]
            h_layer,a = self.calculat_layerWeights(h_layer,a)
            
                    
            error = CrossEntropy(a[-1][0],y[i])
            deltas = [error * self.activation_prime(h_layer[-1])]
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(h_layer[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] -= learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: 
                print ('epochs:', k)
                error_log.append(error)
        return error_log
    
    def describe(self):
        for l in range(0, len(self.weights)):
            print("Weight in layer :",l,"is",self.weights[l])
            
    def predict(self, x): 
        a = np.concatenate((np.array([[1]]), np.array([x])), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

nn = NeuralNetwork([2,3,3,3,1])
X,y = create_training_set()
nn.fit(X, y,.1)
count=0
for i,r in enumerate(X):
    result = nn.predict(r)[0][0]
    if result>.5:
        p=1
    else:
        p=0
    if(p==y[i]):
        count+=1
print("Accuracy : ",str(count/100))
print("Hidden Layer weights")
nn.describe()
