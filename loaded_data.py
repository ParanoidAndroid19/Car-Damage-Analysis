import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, MaxPooling2D
import pickle
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))
X1 = pickle.load(open("X'.pickle", "rb"))
Y1 = pickle.load(open("Y'.pickle", "rb"))

X=X/255
X1=X1/255
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
model.fit(X,Y, batch_size=5,epochs=9, validation_split=0.1)
test_loss, test_acc = model.evaluate(X1, Y1)
print('test acc= ',test_acc)
print('test loss= ',test_loss)
predictions=model.predict(X1)
print(predictions[6])
print('compare')
print(Y1[6])
model.save('untitled.model')
