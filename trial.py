import cv2, numpy as np
import tensorflow as tf
categories= ["00_front_minor", "01_front_moderate", "02_front_major", "03_rear_minor",
             "04_rear_moderate", "05_rear_major", "06_side_minor", "07_side_moderate",
             "08_side_major", "09_whole"]

IMG_SIZE = 150


def prepare(filepath):
    img_arr=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_arr=cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
    return new_arr/255

model=tf.keras.models.load_model("untitled.model")
print(model)
X = [prepare('3.jpg')]
X=np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# print(p)
prediction= model.predict(X)
a = np.sum(prediction[0])
print("\nDamage Category:")
print(categories[np.argmax(prediction[0])])
print(a)
print("\n\n\n")
