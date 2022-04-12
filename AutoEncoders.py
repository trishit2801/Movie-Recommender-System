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


# In[2]:


training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
print(training_set)
training_set = np.array(training_set, dtype = 'int')  # converting the dataframe into a numpy array
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
print(test_set)
test_set = np.array(test_set, dtype = 'int')  # converting the dataframe into a numpy array


# In[ ]:


# Getting the total number of users and movies


# In[3]:


nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# In[4]:


print("Total users: ",nb_users)
print("Total movies: ",nb_movies)


# In[ ]:


# Converting the data into a matrix with 'users' in rows and 'movies' in columns (usual structure for any deep learning model)
# We will create a list of list containing 943 lists of users where each list contains the ratings of 1682 movies


# In[5]:


def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_users]  # extracts all the movie ids of the current user
        id_ratings = data[:, 2][data[:, 0] == id_users] # extracts all the ratings of the current user
        ratings = np.zeros(nb_movies)  # initialising a list of 1682 0s
        ratings[id_movies - 1] = id_ratings  # list belonging to current user gets updated by ratings of movies which are rated by the current user. Movies which are not rated by current user are rated as 0.
        new_data.append(list(ratings))  # adding the list belonging to single user to the list of list. This way 943 lists get added to list of list
    return new_data


# In[6]:


training_set = convert(training_set)  
test_set = convert(test_set)


# In[ ]:


# Converting the data into Torch tensors


# In[7]:


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# In[ ]:


# Creating the architecture of the Neural Network (Stacked AutoEncoders)


# In[8]:


class SAE(nn.Module):  # Here nn.Module is the base class for neural networks. We are creating a subclass SAE that extends this nn.Module class
    def __init__(self, ):
        super(SAE, self).__init__()   # To get all the inherited classes and methods of nn module class
        self.fc1 = nn.Linear(nb_movies, 20)  # Full connection between input layer and 1st hidden layer created using Linear class of nn module. No.of neurons in each hidden layer can be taken anything (Refer PyTorch Documentation for further details)
        self.fc2 = nn.Linear(20, 10)  # Full connection between 1st hidden layer and 2nd hidden layer
        self.fc3 = nn.Linear(10, 20)  # Full connection between 2nd hidden layer and 3rd hidden layer
        self.fc4 = nn.Linear(20, nb_movies)  # Full connection between 3rd hidden layer and output layer (output layer has same dimension as input layer in AutoEncoders)
        self.activation = nn.Sigmoid()   # Taking the activation func as 'sigmoid'. Here we use 'Sigmoid' class of nn module.
    def forward(self, x):   # method for performing operations inside the SAE, i.e. to perform encoding and decoding (Forward Propagation)
        x = self.activation(self.fc1(x))  # 1st encoding for 1st full connection
        x = self.activation(self.fc2(x))  # 2nd encoding for 2nd full connection
        x = self.activation(self.fc3(x))  # 1st decoding for 3rd full connection
        x = self.fc4(x)   # 2nd (Final) decoding for 4th full connection
        return x    # Now 'x' becomes the vector of predicted ratings
    
sae = SAE()
criterion = nn.MSELoss()  # defining the loss function (Mean squared Error) using MSELoss class of nn module
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)   # defining the Optimizer using RMSprop class of optim module
# weight_decay is used to reduce the lr after every few epochs. This improves the model


# In[ ]:


# Training the SAE


# In[9]:


nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)  # adding a second dimension (second dimension is of batch)to the input as PyTorch cannot take single dimension (just like keras)
        target = input.clone()  # creating a backup variable for input vector
        if torch.sum(target.data > 0) > 0:   # considering those users who have rated atleast 1 movie
            output = sae.forward(input)  # calling the forward() to get the predicted ratings
            target.require_grad = False  # for optimizing the code to reduce a lot of computations by not calculating the gradient
            output[target == 0] = 0   # The ratings which are originally 0 (not rated by a user) are taken as 0 in final output
            loss = criterion(output, target)  # calculating the loss by comparing the predicted ratings and actual ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()  # Performing Backpropagation for adjusting the weights. Determines whether we need to increase the weights or decrease the weights
            train_loss += np.sqrt(loss.item() * mean_corrector) # calculating the Root Mean Square Error (RMSE)
            s += 1.
            optimizer.step()  # To apply the optimizer of RMSprop class we use inbuilt step() of the class
            # backwards() decides whether weights are to be increased or decreased whereas optimizers decides by how much amount the weights are to be adjusted
    print('epoch: '+str(epoch)+' train_loss: '+str(train_loss/s))


# In[ ]:


# Testing the SAE


# In[11]:


test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)  # adding a second dimension (second dimension is of batch)to the input as PyTorch cannot take single dimension (just like keras)
    # here we take training set and not test set because we want to predict the ratings of the movies that the user has not watched in the training set and then compare these predicted ratings with the actual ratings of those movies that are present in the test set.
    target = Variable(test_set[id_user]).unsqueeze(0)  # target contains the actual ratings of the movies in the test set that were not watched by the users in the training set
    if torch.sum(target.data > 0) > 0:   # considering those users who have rated atleast 1 movie
        output = sae.forward(input)  # calling the forward() to get the predicted ratings
        target.require_grad = False  # for optimizing the code to reduce a lot of computations by not calculating the gradient
        output[target == 0] = 0   # The ratings which are originally 0 (not rated by a user) are taken as 0 in final output
        loss = criterion(output, target)  # calculating the loss by comparing the predicted ratings and actual ratings
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector) # calculating the Root Mean Square Error (RMSE)
        s += 1.
print('test_loss: '+str(test_loss/s))


# In[ ]:


# Making Predictions for a given user and for a given movie


# In[70]:


user_id = 3
movie_id = 482
input = Variable(training_set[user_id-1]).unsqueeze(0)
predicted_rating = sae.forward(input)
predicted_rating = predicted_rating.data.numpy()
print('Predicted Rating: '+ str(predicted_rating[0, movie_id-1]))


# In[ ]:




