#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd


# In[28]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from genetic_algorithm import GeneticAlgorithm # assume genetic_algorithm is a module containing the genetic algorithm code


# In[29]:


# Load the dataset
df = pd.read_csv(r"C:\Users\satv\Downloads\Bank_Personal_Loan_Modelling.csv")


# In[30]:


# Split the data into features and labels
X = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1).values
y = df['Personal Loan'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[32]:


# Define the neural network architecture
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[33]:


# Define the fitness function
def fitness_function(chromosome):
    model.set_weights(chromosome)
    score = model.evaluate(X_train, y_train, verbose=0)[1]
    return score

# Define the genetic algorithm parameters
pop_size = 20
num_generations = 100
mutation_rate = 0.1


# In[38]:


# Create the genetic algorithm object
ga = GeneticAlgorithm(fitness_function, model.get_weights(), pop_size, mutation_rate)


# In[39]:


best_weights = ga.run(num_generations)


# In[ ]:


model.set_weights(best_weights)
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print("Accuracy:", accuracy)

