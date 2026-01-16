"""
File:           ModelFitter-Example2.py
Description:    A demontration example on usage of the ModelFitter class.               
Author:         Dashan Chang
Created:        1/16/2026
"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from ModelFitter import ModelFitter, curve_fits
import time


def model_sin(x, *p):
    a = p[0]
    b = p[1]
    c = p[2]
    f = a * np.sin(b * x) + c * x
    return f

# nonlinear function
X = np.linspace(0.1, 10, 100)
y = 5 * np.sin(2 * X) + X

noise = np.random.normal(0, 2, len(X))
y = y + noise

print(len(X))
fig, ax = plt.subplots()
ax.plot(X, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
fig,ax=plt.subplots()
ax.scatter(X_train,y_train)

x = X_train
Y = y_train
p = np.array([10, 2.2, 10])
model = model_sin

start_time = time.perf_counter()
sigma = []
popt, pcov = curve_fits(model, x, Y, p)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The code block executed in {elapsed_time:.4f} seconds")

x = np.linspace(0.1, 10, 1000)
y = model(x, *popt)
ax.scatter(x, y)
plt.show()
