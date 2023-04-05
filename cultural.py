#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[16]:



# Load the dataset
df = pd.read_csv(r'C:\Users\satv\Downloads\Bank_Personal_Loan_Modelling.csv')


# In[17]:



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(["ID", "Personal Loan"], axis=1),
    data["Personal Loan"],
    test_size=0.2,
    random_state=42,
)


# In[18]:



# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[19]:



# Define the fitness function
def fitness_function(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]


# In[20]:



# Define the cultural algorithm
population_size = 10
num_generations = 10
alpha = 0.1
beta = 0.1

population = []
fitness_values = []

for i in range(population_size):
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    population.append(model)
    fitness_values.append(fitness_function(model, X_train, y_train, X_test, y_test))

for generation in range(num_generations):
    # Cultural transmission
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    elite_population = sorted_population[:int(alpha*population_size)]
    diverse_population = sorted_population[int(alpha*population_size):]

    # Individual learning
    for model in diverse_population:
        mutation = model.get_weights()
        for layer in mutation:
            layer += beta * np.random.randn(*layer.shape)
        mutated_model = Sequential()
        mutated_model.add(Dense(32, input_dim=X_train.shape[1], activation="relu"))
        mutated_model.add(Dense(16, activation="relu"))
        mutated_model.add(Dense(1, activation="sigmoid"))
        mutated_model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
        mutated_model.set_weights(mutation)
        population.append(mutated_model)
        fitness_values.append(fitness_function(mutated_model, X_train, y_train, X_test, y_test))

    # Environmental selection
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    population = sorted_population[:population_size]
    fitness_values = fitness_values[:population_size]


# In[21]:



# Evaluate the best model
best_model = population[0]
accuracy = fitness_function(best_model, X_train, y_train, X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


data = pd.read_csv(r"C:\Users\LEN\Downloads\new_test (1).csv")
data.drop(["contact","month","day_of_week","duration","campaign","poutcome"] , axis = 1,inplace=True)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
x = torch.tensor(x , dtype = torch.float64)
y = torch.tensor(y , dtype=  torch.float64)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25)


# In[7]:


class nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(7, 20 )
        self.linear2 = torch.nn.Linear(20, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x


# In[8]:


model = nn()
loss_function = torch.nn.CrossEntropyLoss()


# In[9]:


class ga:
    def __init__(self, model, population_size, mutation , decay ,  inputs  , labels):
        self.model = model
        self.population_size = population_size
        self.mutation = mutation
        self.population = self.init_population()
        self.decay = decay
        self.inputs = inputs
        self.labels = labels

    def init_population(self):
        population = []
        for i in range(self.population_size):
            weights = []
            for weight in self.model.parameters():
                weights.append(weight.data.numpy())
            population.append(weights)
        return population

    def selection(self, fitness_scores):
        cumulative_scores = np.cumsum(fitness_scores)
        total_score = np.sum(fitness_scores)
        rand = np.random.uniform(0, total_score)
        selected_index = np.searchsorted(cumulative_scores, rand)
        return selected_index

    def crossover(self, male, female):
        random_crossover = np.random.randint(1, len(male))
        child1 = male[:random_crossover] + female[random_crossover:]
        child2 = male[:random_crossover] + female[random_crossover:]
        return child1, child2
    
    def decay_mutation_rate(self):
        self.mutation -= (self.decay*self.mutation)

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.uniform(0, 1) < self.mutation:
                child[i] += np.random.normal(0, 0.1, child[i].shape)
        return child

    def generate_offspring(self, fitness_scores):
        new_population = []
        for _ in range(self.population_size):
            parent1_index = self.selection(fitness_scores)
            parent2_index = self.selection(fitness_scores)
            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def update_weight(self):
        fitness_scores = [self.fitness(weights) for weights in self.population]
        best_index = np.argmax(fitness_scores)
        best_weights = self.population[best_index]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

    def fitness(self, weights):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(weights[i])
        outputs = self.model(self.inputs)
        loss = loss_function(outputs.float(), self.labels.reshape([len(self.inputs) , 1]).float())
        return 1 / (loss.item() + 1e-6)


# In[13]:


GA = ga(model, population_size=20, mutation=0.3  , decay = 0.03, inputs = x_train, labels = y_train)


# In[14]:


loss_list = []
n=10
for epoch in range(n):
            GA.generate_offspring([])
            GA.update_weight()
            outputs = model(x_train)
            loss = loss_function(outputs, y_train.reshape([len(x_train) , 1]).float())
            loss_list.append(loss.item())
            loss.backward()
            GA.generate_offspring([])
            GA.update_weight()
            if (epoch%10 == 0):
                print("Epoch" , epoch , " : " , loss.item());
                GA.decay_mutation_rate()
print(loss_list)
    


# In[ ]:




