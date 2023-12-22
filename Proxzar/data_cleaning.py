#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import os
import shutil
import pytesseract


# In[20]:


dataset_path = r"C:\Users\Administrator\Downloads\becnprods4-products-images"
destination_path = r'C:\Users\Administrator\Downloads\Removed-pics'


# In[21]:


# Function to check if an image contains text using pytesseract OCR
def contains_text(image_path, threshold=5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # You can adjust the threshold based on your requirements
    return len(text) > threshold


# In[22]:


# Iterate through images in the dataset folder
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
        image_path = os.path.join(dataset_path, filename)

        # Check if the image contains text
        if contains_text(image_path):
            # Move the image file to the destination folder
            shutil.move(image_path, os.path.join(destination_path, filename))
            print(f"Moved text-containing image: {filename}")


# In[ ]:




