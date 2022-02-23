import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing import image
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

TEST_PATH = './data/test/'
MODEL_PATH = './data/weight/model_eff.h5'

IMG_SIZE = 380
IMG_SHAPE = (380, 380, 3)

test_data = []
images_list = []
for img in os.listdir(TEST_PATH):
    images_list.append(img)
    img = os.path.join(TEST_PATH, img)
    img = image.load_img(img, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    test_data.append(img)

test_data = np.vstack(test_data)
#test_data = test_data/255.

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.0003),
    metrics=["acc"],
)

prediction = np.argmax(model.predict(test_data), axis=-1)

submit = pd.DataFrame({
                 'image_name': images_list,
                 'class_id': prediction        
                })

submit.to_csv('./data/out/submission.csv',  index=False, sep='\t')