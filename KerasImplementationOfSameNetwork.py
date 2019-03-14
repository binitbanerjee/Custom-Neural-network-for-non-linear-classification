import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
from keras import optimizers

def create_training_set():
    class1=[]
    class2=[]
    label = [1,1,0,0]
    net = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    
    for i in range(24):
        class1.append([np.random.uniform(low=-3,high=-1.1),np.random.uniform(low=0.1,high=1)])
        class2.append([np.random.uniform(low=0,high=.9),np.random.uniform(low=-3,high=-1)])
    
    for i in range(24):
        class1.append([np.random.uniform(low=1.1,high=3),np.random.uniform(low=0.1,high=.99)])
        class2.append([np.random.uniform(low=0,high=.9),np.random.uniform(low=1.1,high=3)])
    net.extend(class1)
    net.extend(class2)
    
    for i in net[4:]:
        if abs(i[0])>1 and abs(i[1])<1:
            label.append(1)
        else:
            label.append(0)
    return np.array(net),np.array(label)


training_data,target_data = create_training_set()

model = Sequential()
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.2)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history =  model.fit(training_data, target_data, nb_epoch=10,steps_per_epoch=1000, verbose=1)
output  =  (model.predict(training_data).round())
count = 0

for i,r in enumerate(output):
  if(target_data[i]==r):
    count+=1

print(count/100)
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(output)
