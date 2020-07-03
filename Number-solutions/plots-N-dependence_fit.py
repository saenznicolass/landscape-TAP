# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 01:22:17 2020

@author: n904
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.optimize import curve_fit
import csv

estimado=[7,8,10,14,17,24,25,43,56,86,110,228,596,1816]
#estimado=[64,128,236,508,724,1504,3388,5628,12542,28804,44444,76618,166632,388312]
particulas=[6,7,8,9,10,11,12,13,14,15,16,17,18,19]

for i in [6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
    data = pd.read_csv('{}-total/total.csv'.format(i))

    x=list(data['energia'])
    y=list(data['parametro'])
    x_filtr=x
    print('N:', i,"total:", len(x), 'filtrado:', len(x_filtr))
    
    



def func(x, b):
  return np.exp(b*0.1*x)
  #return a * np.log(b * x) + c

x1 = particulas
x1 = np.array(x1, dtype=float)  # changed boundary conditions to avoid division by 0
y1 = np.array(estimado, dtype=float)

# popt, pcov = curve_fit(func, x1, y1)
# with open("parameters_fit.csv", "a", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(zip(['a'],popt))



# FIG=plt.figure(figsize=(10,7))
# plt.plot(x1, y1, 'ko', markersize=3,label="TAP")
# plt.plot(x1, func(x1,popt[0]), label="Fitted")
# plt.xlabel('N')
# plt.ylabel('# TAP')
# plt.legend()
# plt.xlim(5,20)
# plt.show()
# FIG.savefig('Dependence-N-b-T=0.1.png')



# def func2(x,a,b):
#   return a*np.exp(b*0.1*x)
#   #return a * np.log(b * x) + c

# x2 = particulas
# x2 = np.array(x2, dtype=float)  # changed boundary conditions to avoid division by 0
# y2 = np.array(estimado, dtype=float)

# popt, pcov = curve_fit(func2, x2, y2)

# with open("parameters_fit.csv", "a", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(zip(['a','b'],popt))


# FIG=plt.figure(figsize=(10,7))
# plt.plot(x2, y2, 'ko', markersize=3,label="TAP")
# plt.plot(x2, func2(x1, popt[0], popt[1]), label="Fitted")
# plt.xlabel('N')
# plt.ylabel('# TAP')
# plt.legend()
# plt.xlim(5,20)
# plt.show()
# FIG.savefig('Dependence-N-a-b-T=0.1.png')




def func3(x,a,b,c):
  return a*np.exp(b*0.1*x) + c
  #return a * np.log(b * x) + c

x3 = particulas
x3 = np.array(x3, dtype=float)  # changed boundary conditions to avoid division by 0
y3 = np.array(estimado, dtype=float)
x4=np.linspace(6,19,100)
popt, pcov = curve_fit(func3, x3, y3)

with open("parameters_fit.csv", "a", newline='') as f:
    wr = csv.writer(f)
    wr.writerow(zip(['a','b','c'],popt))


FIG=plt.figure(figsize=(12,7))
plt.plot(x3, y3, 'ko', markersize=3,label="TAP")
plt.plot(x4, func3(x4, popt[0], popt[1],popt[2]), label="Fitted")
plt.xlabel('N',size = 20)
plt.ylabel('# TAP',size = 20)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlim(5,20)
plt.show()
FIG.savefig('Dependence-N-a-b-c-T=0.1.png')

print(func3(20, popt[0], popt[1],popt[2]))