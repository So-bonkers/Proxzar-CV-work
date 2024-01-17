#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# In[26]:


class ClassToken(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


# In[27]:


def mlp(x, cf):
    x = Dense(cf['mlp_dim'], activation="gelu")(x)
    x = Dropout(cf['dropout_rate'])(x)
    x = Dense(cf['hidden_dim'])(x)
    x = Dropout(cf['dropout_rate'])(x)
    return x


# In[31]:


def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(cf['num_heads'], key_dim=cf['hidden_dim'])(x,x)
    x = Add()([ x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x,cf)
    x = Add()([x, skip_2])

    return x


# In[32]:


def ViT(cf):
    # Inputs
    input_shape = (cf["num_patches"],cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = Input(input_shape)
    # print(inputs)

    #  Patch Embedding and Positional Embedding
    patch_embed = Dense(cf["hidden_dim"])(inputs)
    # print(patch_embed)
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)
    # print(pos_embed.shape)
    embed = patch_embed + pos_embed # the input to the ViT is a sum of patch embedding and positional embedding
    # print(embed.shape)

    # Adding class token
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed])
    # print(x.shape)
    
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)
    # print(x.shape) ## output of the entire encoder block
    
    
    """Classfication Head"""


# In[33]:


if __name__ == "__main__":
    config={}
    config['num_layers'] = 12
    config['num_heads'] = 12
    config['hidden_dim'] = 768
    config['dropout_rate'] = 0.1
    config['mlp_dim'] = 3072
    config["num_patches"] = 256
    config["patch_size"] = 32
    config["num_channels"] = 3
    
    ViT(config)


# In[ ]:




