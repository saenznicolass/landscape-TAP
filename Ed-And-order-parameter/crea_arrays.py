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

N=20
n=20



aguardar=[]
for i  in range(20000):
    mag=np.random.random((N, 3))*2-1
    aguardar.append(mag)

b=np.array_split(aguardar, n)

# print("----------")
# print(aguardar)

# print(b)
for i in range(len(b)):
    np.save("list_qea_inter_free_N={}_array_{}.npy".format(N,i),b[i])

#mag_list=np.load("lista_para_N={}_array_{}.npy".format(N,n-1))
#print(mag_list)