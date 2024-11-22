#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[71]:


# Generate dataset
X, y = make_regression(n_features=1, n_samples=100, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
X_train = X_train.T
y_train = y_train.reshape(1, -1)
X_test = X_test.T
y_test = y_test.reshape(1, -1)


# In[58]:


# Initialize weights and bias
def initialize(X):
    w = np.random.randn(1, X.shape[0])
    b = np.random.randn(1)
    return (w, b)


# In[59]:


# Predict function
def predict(w, b, X):
    return w.dot(X) + b


# In[60]:


# Loss function
def loss_function(y_pred, y):
    return np.mean((y_pred - y) ** 2)


# In[61]:


# Gradients calculation
def gradients(X_train, w, b, y_train):
    y_pred = predict(w, b, X_train)
    dw = (2 / y_train.shape[1]) * np.dot(X_train, (y_pred - y_train).T).T
    db = (2 / y_train.shape[1]) * np.sum(y_pred - y_train)
    return dw, db


# In[62]:


# Update weights and bias
def update(w, b, dw, db, p):
    w = w - p * dw
    b = b - p * db
    return w, b


# In[63]:


# Linear regression function
def regression_lineaire(X_train, y_train, w, b, p, n_iter):
    losses = []
    for i in range(n_iter):
        dw, db = gradients(X_train, w, b, y_train)
        w, b = update(w, b, dw, db, p)
        losses.append(loss_function(predict(w, b, X_train), y_train))
    plt.plot(losses)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    return w, b


# In[56]:


# Main execution
w, b = initialize(X_train)
w, b = regression_lineaire(X_train, y_train, w, b, p=0.1, n_iter=10)

# Final plot with regression line
plt.scatter(X, y, label="Data Points")  # Scatter plot of original data
plt.plot(X, predict(w, b, X.T).flatten(), color='red', label="Regression Line")  # Single regression line
plt.title("Regression Line Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




