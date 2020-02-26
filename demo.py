import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os,cv2
import pickle
data_dir= r"/Users/simranbhake/Desktop/Car_Damage"
categories= ["00_front_minor", "01_front_moderate", "02_front_major", "03_rear_minor",
             "04_rear_moderate", "05_rear_major", "06_side_minor", "07_side_moderate",
             "08_side_major", "09_whole"]
img_size=150
training_data=[]
def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num =categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr, (img_size, img_size))
                training_data.append([new_arr, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))
random.shuffle(training_data)
X=[]
Y=[]
for features, labels in training_data:
    X.append(features)
    Y.append(labels)

X=np.array(X).reshape(-1, img_size, img_size, 1)
pickle_out=open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out=open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)
print(X[1])
print(Y)
print(len(X))
