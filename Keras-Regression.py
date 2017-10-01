import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# generation some house sizes between 1000 and 3500 (typical sq. ft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
# print(house_size)

# generate the house price based on house sizes with a random noise added.
np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)
# print(house_price)

# Let us plot the house vs size.
#plt.plot(house_size, house_price, "bx") # bx is blue x
#plt.xlabel("Size")
#plt.ylabel("Price")
#plt.show()

def normalize(array):
    return ((array - array.mean()) / array.std())

# Let us 70% of the data set as the training set for the algorithm 
num_train_samples = math.floor(0.7 * num_house)

###### Step 1 - Data Preparation - We are defining the training data set and test data set from the given dataset.

# Define the Training data based on the num_train_samples
train_house_size = house_size[:num_train_samples]
train_house_price = house_price[:num_train_samples]

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Define the Test data 
test_house_size = house_size[num_train_samples:]
test_house_price = house_price[num_train_samples:]

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)


# There are 4 steps to prepare before we execute the tensor flow graph.
# 1. Prepare the data - In this case it is the test and training dataset.
# 2. Define the inference function - In this case the inference function is the predicate that predicts the price based on the size and size factor and the price offset
# 3. Define the Loss calculation - The losss calculation is defined using the above predicate.
# 4. Finally the Optimizer - This optimizer is what we want to optimize based on the loss function.


# Define the NN for doing Linear Regression
model = Sequential()
model.add(Dense(1, input_shape=(1,), init='uniform', activation='linear'))

model.compile(loss="mean_squared_error", optimizer='sgd') # Loss and Optimizer

# Fit or Train the model
model.fit(train_house_size_norm, train_house_price_norm, nb_epoch=300)

# Evaluate with the test data
score = model.evaluate(test_house_size_norm, test_house_price_norm)
print("\n Loss on test: {0}".format(score))

