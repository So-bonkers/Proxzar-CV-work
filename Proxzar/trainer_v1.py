#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras import layers
from keras import models, optimizers
import matplotlib.pyplot as plt
from PIL import Image
from PIL import UnidentifiedImageError  # Import UnidentifiedImageError explicitly
from vit import ViT


# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


# Hyperparameters dictionary
hp = {}
hp["image_size"] = 256
hp["num_channels"] = 3 
hp["patch_size"] = 32
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 64
hp["learning_rate"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 7
hp["class_names"] = ["gutters","roofing","insulation","waterproofing","tools-equipment","siding","building-materials"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1


# In[4]:


print("hp keys:", hp.keys())  # Check all keys in hp
print("Batch size:", hp.get("batch_size"))  # Check the value associated with "batch_size"
print("Flat patches shape: ", hp.get("flat_patches_shape"))
print("Number of patches: ", hp.get("num_patches"))


# In[5]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[6]:


def load_data(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)

    return train_x, valid_x, test_x


# In[7]:


def process_image_label(path):
    # Reading Images
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image/255.0
    
    # print("Image shape is: ", image.shape)
    # cv2.imwrite(f"files/original_image.jpg", image)

    # Preprocessing to patches
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])

    # patches = np.reshape(patches, (64,32,32,3))
    # print("Patches shape is: ", patches.shape)
    # for i in range(64):
    #     cv2.imwrite(f"files/{i}.png", patches[i])

    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    # Label
    # print(path)
    class_name = path.split("/")[-2]
    # print("Class name: ",class_name)
    class_index = hp["class_names"].index(class_name)
    class_index = np.array(class_index, dtype=np.int32)
    # print("Class Index: ", class_index)

    return patches, class_index



# In[8]:


def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp["num_classes"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])

    return patches, labels


# In[9]:


def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds


# In[10]:


if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing files
    create_dir("files")

    # Paths
    dataset_path = "/home/sksystem/Downloads/Proxzar-CV-work-main/Final_folder_for_classification"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    # Dataset
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    # process_image_label(train_x[0])

    
    train_ds = tf_dataset(train_x, batch = hp["batch_size"])
    valid_ds = tf_dataset(valid_x, batch = hp["batch_size"])

    for x, y in train_ds.take(1):
        print(x.shape)
        print(y.shape)

    # Model
    model = ViT(hp)
    model.compile(
        loss="categorical_crossentropy", # Because its a multi-class classification
        optimizer = tf.keras.optimizers.Adam(hp["learning_rate"], clipvalue=1.0),
        metrics=["acc"]
    )
    
    callbacks = [
    ModelCheckpoint(filepath="files/model_epoch_{epoch:02d}.h5", verbose=1, save_best_only=True, save_freq='epoch', period=10),
    CSVLogger(csv_path),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]



    # Train the model
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=hp["num_epochs"],
        callbacks=callbacks,
        verbose=1
    )
    
    # # Train the model with tqdm progress bar
    # with tqdm(total=hp["num_epochs"], desc="Training") as pbar:
    #     for epoch in range(hp["num_epochs"]):
    #         model.fit(
    #             train_ds,
    #             epochs=1,  # Train for 1 epoch at a time
    #             validation_data=valid_ds,
    #             verbose=0  # Set verbose to 0 to suppress TensorFlow's progress bar
    #         )
    #         pbar.update(1)  # Update tqdm progress bar for each epoch


# In[ ]:




