# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 05:38:45 2020

crear todas las diferentes magnetizaciones iniciales

@author: Boris Saenz
"""
from itertools import product
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N=12
n=2

a=list(product([1,-1], repeat=N))


aguardar=[]
for elem in a:
    mag=np.random.random((N, 3))*2-1
    mag[:,:]=np.array(elem).reshape(N,1)
    aguardar.append(mag)

b=np.array_split(aguardar, n)

# print("----------")
# print(aguardar)

# print(b)
for i in range(len(b)):
    np.save("lista_para_N={}_array_{}.npy".format(N,i),b[i])

#mag_list=np.load("lista_para_N={}_array_{}.npy".format(N,n-1))
#print(mag_list)