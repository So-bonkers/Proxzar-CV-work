#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os
import pandas as pd


# In[48]:


dataset_path = r"C:\Users\Administrator\Downloads\proxzar_shubhankar\Aug_images\P25 - Visual Search\Augmented images"


# In[49]:


categories =[]
subtypes=[]
fileid=[]
filenames =[]


# In[50]:


for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        file_parts = filename.split("_")
        if len(file_parts) >=2:
            category = file_parts[0]
            subtype = file_parts[1]
            fileids = file_parts[2]
            fileid.append(fileids)
            subtypes.append(subtype)
            categories.append(category)
            filenames.append(filename)


# In[51]:


fileid = [fileid.replace(".jpg", "") for fileid in fileid]

# In[52]:


total_unique_categories = len(set(categories))
print(total_unique_categories)


# In[53]:


data = {"Category": categories, "Sub-Category": subtypes, "Product": fileid, "Filename": filenames}
df = pd.DataFrame(data)


# In[54]:


# df


# In[55]:


roofing_subtypes = df[df["Category"] == "roofing"]["Sub-Category"].unique()
len(set(roofing_subtypes))


# In[56]:


df.to_csv('categories_filenames.csv', index=False)


# In[59]:


# # Load data from the CSV file
# data = pd.read_csv('/home/sklaptop/Downloads/Proxzar-CV-work/Proxzar/categories_filenames.csv')
#
# df = pd.DataFrame(data)
#
#
# In[60]:


category_counts = df['Category'].value_counts()
print(category_counts)


# In[ ]:




