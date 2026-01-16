"""
File:           ModelFitter-Example1.py

Description:    A demontration example on usage of the ModelFitter class. 
                The antelope population model appears on many text book on numerical algorithm

Author:         Dashan Chang

Created:        1/16/2026

"""
import numpy as np
import matplotlib.pyplot as plt
from ModelFitter import ModelFitter, curve_fits

def model_antelope_population(x, *p):
    a = p[0]
    k = p[1]
    f = a * np.exp(k * x)
    return f

T = np.array([1,2,4,5,8])
Y = np.array([3,4,6,11,20])


P = np.array([2, 1])
model = model_antelope_population
sigma = np.array([1., 1., 1., 1., 1.])

popt, pcov = curve_fits(model, T, Y, p0=P, sigma=sigma)

print("Optimized parameters:")
print(np.round(popt, decimals=8))
print("Variance and covariance of parameters:")
print(pcov)

plt.scatter(T,Y,label='Data')
t = np.linspace(1,10,100)
y = model(t, *popt)
plt.plot(t, y, 'b-', label='Fit')
plt.scatter(T,Y,label='Data')
plt.show()

print("Measured Y[i]:", Y)
y = model(T, *popt)
print ("Predicted y[i]:", y)
loss = Y - y
print("loss[i]:", loss)
ss = loss ** 2
print("square of loss[i]]:", ss)
s = np.sqrt(ss.sum()/len(Y))
print("Square Root of the sum of squares:", s)