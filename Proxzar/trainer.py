import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from vit import ViT

# Hyperparameters dictionary
hp = {}
hp["image_size"] = 256
hp["num_channels"] = 3
hp["patch_size"] = 16
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_images"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 32
hp["learning_rate"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 7
hp["class_names"] = ["gutters","roofing","insulation","waterproofing","tools-equipment","siding","building-materials"]
