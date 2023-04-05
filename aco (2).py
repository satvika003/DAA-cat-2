#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[33]:


df = pd.read_csv(r'C:\Users\satv\Downloads\Bank_Personal_Loan_Modelling.csv')


# In[34]:


# Preprocess the data
X = df.iloc[:, 1:12]
y = df.iloc[:, -1]
X = pd.get_dummies(X, columns=['Education', 'Family', 'Securities Account', 'CD Account', 'Online', 'CreditCard'])
X = (X - X.mean()) / X.std()
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


model = Sequential()
model.add(Dense(10, input_dim=11, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[36]:


def fitness_function(weights):
    model.set_weights(weights)
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    return accuracy


# In[37]:


def ant_colony_optimization(population_size, alpha, beta, evaporation_rate, generations):
    pheromone_trails = np.ones(model.get_weights())
    best_fitness = 0
    best_weights = []
    for generation in range(generations):
        population = generate_ants(population_size, pheromone_trails)
        fitness_scores = [fitness_function(weights) for weights in population]
        best_ant = population[np.argmax(fitness_scores)]
        if fitness_scores[np.argmax(fitness_scores)] > best_fitness:
            best_fitness = fitness_scores[np.argmax(fitness_scores)]
            best_weights = best_ant
        pheromone_trails = update_pheromones(pheromone_trails, population, fitness_scores, alpha, beta, evaporation_rate)

    model.set_weights(best_weights)
    test_accuracy = fitness_function(best_weights)
    return test_accuracy


# In[38]:


def ant_colony_optimization(population_size, alpha, beta, evaporation_rate, generations):
    pheromone_trails = np.ones(model.get_weights())
    best_fitness = 0
    best_weights = []
    for generation in range(generations):
        population = generate_ants(population_size, pheromone_trails)
        fitness_scores = [fitness_function(weights) for weights in population]
        best_ant = population[np.argmax(fitness_scores)]
        if fitness_scores[np.argmax(fitness_scores)] > best_fitness:
            best_fitness = fitness_scores[np.argmax(fitness_scores)]
            best_weights = best_ant
        pheromone_trails = update_pheromones(pheromone_trails, population, fitness_scores, alpha, beta, evaporation_rate)

    model.set_weights(best_weights)
    test_accuracy = fitness_function(best_weights)
    return test_accuracy


# In[39]:


def generate_ants(population_size, pheromone_trails):
    population = []
    for i in range(population_size):
        weights = []
        for j in range(model.get_weights()):
            layer_weights = np.random.rand(model.get_weights()[j].shape[0], model.get_weights()[j].shape[1])
            weights.append(layer_weights)
        population.append(weights)
    return population


# In[40]:


def update_pheromones(pheromone_trails, population, fitness_scores, alpha, beta, evaporation_rate):
    for i in range(len(population)):
        ant_weights = population[i]
        ant_fitness = fitness_scores[i]
        for j in range(len(ant_weights)):
            layer_weights = ant_weights[j]
            for k in range(layer_weights.shape[0]):
                for l in range(layer_weights.shape[1]):
                    pheromone_trails[j][k][l] = (1 - evaporation_rate) * pheromone_trails[j][k][l] + evaporation_rate * ant_fitness * ant_weights[j][k][l]
    return pheromone_trails


# In[41]:


test_accuracy = ant_colony_optimization(population_size=10, alpha=1, beta=1, evaporation_rate=0.5, generations=10)


# In[ ]:





# In[ ]:





# In[ ]:




