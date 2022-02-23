import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

TRAIN_PATH = './data/train/'
CSV_PATH = './data/train.csv'
IMG_SIZE = 380
IMG_SHAPE = (380, 380, 3)


traindf = pd.read_csv(CSV_PATH, sep='\t')

traindf['class_id'] = traindf['class_id'].astype(str)

#datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
datagen = ImageDataGenerator(validation_split=0.2)

train_data=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=TRAIN_PATH,
    x_col="image_name",
    y_col="class_id",
    subset="training",
    batch_size=16,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMG_SIZE,IMG_SIZE))

valid_data=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=TRAIN_PATH,
    x_col="image_name",
    y_col="class_id",
    subset="validation",
    batch_size=16,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMG_SIZE,IMG_SIZE))

base_model = tf.keras.applications.EfficientNetB4(include_top=False,
                                                 weights='efficientnetb4_notop.h5',
                                                 input_shape=IMG_SHAPE)

base_model.trainable = False

model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.0003),
    metrics=["acc"],
)

history = model.fit(train_data,
                    epochs=3,
                    validation_data=valid_data)

model.save('./data/weight/model_eff.h5')