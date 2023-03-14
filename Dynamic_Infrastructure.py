#!/usr/bin/env python
# coding: utf-8

# # Dynamic Infrastructure - Home Assignment
# 
# The goal of this assignment is to evaluate basic scientific data manipulation and library research work. It would take 1-2 hours at most. This notebook contains all the information required for this exercise and the solution should be also written in this notebook.
# 
# ### Part 1: Scientific computing in NumPy
# 
# Your code should be efficient and readable. It is suggested that you use NumPy, however alternatives (i.e. numba) are welcome.

# In[1]:


import numpy as np
import pandas as pd
import re

np.random.seed(42)
X = np.random.randint(low=0, high=50, size=(30, 4))


# In[2]:


X


# 1. Calculate the mean values of each column in X.

# In[3]:


# Your code here

np.mean(X, axis=0)


# 2. Calculate the max value of each row in X.

# In[4]:


# Your code here

np.mean(X, axis=1)


# 3. Calculate the median value of all the values of X.

# In[5]:


# Your code here

np.median(X)


# 4. Normalize X such that every column will have a mean of 0 and a variance of 1.

# In[6]:


# Your code here

normed = (X - X.mean(axis=0)) / X.std(axis=0)

normed.mean(axis=0)
normed.std(axis=0)


# 5. Given a 1D array Y, calculate the difference between each two elements of Y and save the results in a 2D array. This can be done in a single line using broadcasting.

# In[7]:


Y = np.linspace(1, 10, 10)
# Your code here
Y


# In[8]:


Y.reshape(10,1) 


# In[9]:


Y.reshape(10,1) - Y


# 6. Considering a 10x3 matrix, A. Create a new matrix, B, where all rows in B are rows in A apart from rows that have the same value. For example, [2,2,2] will not be copied to B, however [1,4,4] will be copied to B. Solve this for numerical values only. This can be solved in a single line without explicit loops.

# In[10]:


# Your code here

A = np.random.randint(1,4,(10,3))
A


# In[11]:


A[~np.logical_and.reduce(A[:,1:] == A[:,:-1], axis=1)]


# ### Part 2: Using Pandas
# 

# In[12]:


url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo = pd.read_csv(url, sep="\t")

chipo.head(5)  # view the first 5 entries


# 1. How many items were ordered in total?

# In[13]:


# Your code here

chipo['quantity'].sum()


# 2. What was the total revenue?

# In[14]:


sum(chipo.item_price.str.replace('$','').astype(float) * chipo.quantity)


# 3. How many orders were made?

# In[15]:


# Your code here

chipo['order_id'].nunique()


# 4. Count how many unique rows the dataframe has (i.e. ignore all rows that are duplicates).

# In[16]:


# Your code here

chipo['order_id'].nunique()


# In[17]:


df = pd.DataFrame(
    {
        "From_To": [
            "LoNDon_paris",
            "MAdrid_miLAN",
            "londON_StockhOlm",
            "Budapest_PaRis",
            "Brussels_londOn",
        ],
        "FlightNumber": [10045, np.nan, 10065, np.nan, 10085],
        "Airline": [
            "KLM(!)",
            "<Air France> (12)",
            "(British Airways. )",
            "12. Air France",
            '"Swiss Air"',
        ],
    }
)

df.head()


# 5. Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the column an integer column

# In[18]:


# Your code here


df.FlightNumber = df.FlightNumber.interpolate().astype(int)
df


# 6. The From_To column would be better as two separate columns. Split each string on the underscore delimiter `_` to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame.

# In[19]:


df2 = df.copy()
df2[["From", "To"]] = df.From_To.str.lower().str.split("_", expand=True)
df2 = df2[['FlightNumber', 'Airline', 'From', 'To']]
df2


# 7. In the Airline column, you can see some extra punctuation and symbols have appeared around the airline names. Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

# In[20]:


# Your code here

df2['Airline'] = df2['Airline'].str.replace('[^a-zA-Z\s]', '')
df2

