#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries


# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# In[ ]:


# Importing the dataset


# In[2]:


movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


# In[3]:


movies


# In[4]:


users


# In[5]:


ratings


# In[ ]:


# Preparing the training and test set


# In[6]:


training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
print(training_set)
training_set = np.array(training_set, dtype = 'int')  # converting the dataframe into a numpy array
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
print(test_set)
test_set = np.array(test_set, dtype = 'int')  # converting the dataframe into a numpy array


# In[ ]:


# Getting the total number of users and movies


# In[7]:


nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# In[8]:


print("Total users: ",nb_users)
print("Total movies: ",nb_movies)


# In[ ]:


# Converting the data into a matrix with 'users' in rows and 'movies' in columns (usual structure for any deep learning model)
# We will create a list of list containing 943 lists of users where each list contains the ratings of 1682 movies


# In[9]:


def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_users]  # extracts all the movie ids of the current user
        id_ratings = data[:, 2][data[:, 0] == id_users] # extracts all the ratings of the current user
        ratings = np.zeros(nb_movies)  # initialising a list of 1682 0s
        ratings[id_movies - 1] = id_ratings  # list belonging to current user gets updated by ratings of movies which are rated by the current user. Movies which are not rated by current user are rated as 0.
        new_data.append(list(ratings))  # adding the list belonging to single user to the list of list. This way 943 lists get added to list of list
    return new_data


# In[10]:


training_set = convert(training_set)  
test_set = convert(test_set)


# In[ ]:


# Converting the data into Torch tensors


# In[11]:


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# In[ ]:


# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)  [We are doing this step because in this particular problem we are trying to predict whether a movie shall be liked or not liked by a user]


# In[12]:


training_set[training_set == 0] = -1 # converting all the movie ratings that were not rated by a particular user from 0 to -1, as now 0 (Not Liked) has a different significance

# converting the movies having ratings 1 and 2 to '0'(Not Liked) and movies having ratings 3,4,5 to '1'(Liked)
training_set[training_set == 1] = 0  
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1


# In[13]:


test_set[test_set == 0] = -1 # converting all the movie ratings that were not rated by a particular user from 0 to -1, as now 0 (Not Liked) has a different significance

# converting the movies having ratings 1 and 2 to '0'(Not Liked) and movies having ratings 3,4,5 to '1'(Liked)
test_set[test_set == 1] = 0  
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# In[ ]:


# Creating the architecture of the Neural Network (Restricted Boltzmann Machine model)


# In[21]:


class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)  # initializing random weights to the tensor of size nh(num of hidden nodes) and nv(num of visible nodes)
        self.a = torch.randn(1, nh)  # intializing the bias for hidden nodes
        self.b = torch.randn(1, nv)  # intializing the bias for visible nodes
    def sample_h(self, x):        # calculating P(H=1/V)
        wx = torch.mm(x, self.W.t())  # product of weights and visible nodes(x)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):      # calculating P(V=1/H)
        wy = torch.mm(y, self.W)    # product of weights and hidden nodes(y)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):  # Contrastive Divergence for 'k' steps
        self.W = self.W + (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b = self.b + torch.sum((v0 - vk), 0)
        self.a = self.a + torch.sum((ph0 - phk), 0)
    def predict(self, x):  # x is no.of visible nodes
        _,h = self.sample_h(x)
        _,v = self.sample_v(h)
        return v
# Here 'v0' is the initial node which takes the original ratings and remains same.
# Here 'vk' is the visible node which changes after every step of 'k' steps when weights gets adjusted by train() of Contrastive Divergence.
# Final value of 'vk' is the value after last step of contrastive divergence. It is the predicted value.
# 'v0' is the actual value.


# In[22]:


nv = len(training_set[0])  # no.of features in visible layer = Number of movies = no.of features in first row of training set
nh = 100   # no.of features in hidden layer (can be taken anything)
batch_size = 100    # can be taken anything
rbm = RBM(nv, nh)


# In[ ]:


# Training the RBM


# In[23]:


nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.  # counter which is used to normalise the train_loss
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user: id_user+batch_size]  # creating a batch of 100 users
        v0 = training_set[id_user: id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)  # getting the first value of the return,i.e, p_h_given_v
        for k in range(10):   # Taking 'k' steps as 10
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]  # freezing the visible nodes containing -1 rating (which are not rated by users). We don't want these visible nodes to be included in training. So we freeze these nodes.
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


# In[ ]:


# Testing the RBM


# In[24]:


test_loss = 0
s = 0.  # counter which is used to normalise the train_loss
for id_user in range(nb_users):
    v = training_set[id_user: id_user+1]  # training set is used as an input to activate the hidden neurons of the RBM so that it can predict the test set results
    vt = test_set[id_user: id_user+1]
    if len(vt[vt>=0]) > 0:  # eliminating the -1 ratings (ratings which never happened)
        _,h = rbm.sample_h(v)   # We are taking 1 step as training is already done and now we are evaluating the model on the test set.
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('Test loss: '+str(test_loss/s))


# In[ ]:


# Making Predictions on the movies which have not been seen (or rated) by a user (i.e, for movies having rating -1 for a particular user)


# In[ ]:


# Taking a random user from test data and taking the whole list of ratings of movies of that user and converting it into a 2d Tensor


# In[26]:


user_id = 23
user_input = Variable(test_set[user_id-1]).unsqueeze(0)
prediction = rbm.predict(user_input)
prediction = prediction.data.numpy()


# In[ ]:


# Stack the actual input and our predicted recommendation as one numpy array (Just for comparing the results)


# In[28]:


input_vs_prediction = np.vstack([user_input, prediction])
print(input_vs_prediction)


# In[ ]:




