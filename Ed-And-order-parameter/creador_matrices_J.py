
import matplotlib
matplotlib.use('Agg')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import collections
import numpy as np
import numpy.matlib
import time
import random
import os
import errno

def acoplamientos(N):
    sym=numpy.zeros(shape=(N,N,N))
    standard_deviation = np.sqrt(3)/N
    for i in range (N):
        for j in range (N):
            for k in range (N):
                if ((i-j)*(i-k)*(j-k)!=0): 
#                    J=np.random.normal(loc = mean, scale = standard_deviation)
                    J=np.random.randn()*standard_deviation
#                    print("---------tensor de magnetizaciones---------")
#                    print(i,j,k)
                    sym[i,j,k]=J
                    sym[i,k,j]=J
                    sym[j,i,k]=J
                    sym[j,k,i]=J
                    sym[k,j,i]=J
                    sym[k,i,j]=J
                else:
                    sym[i,j,k] = 0
                    sym[i,k,j] = 0
                    sym[j,i,k] = 0
                    sym[k,j,i] = 0

    #print(sym)
    return sym


mat=acoplamientos(20)
numpy.save("matriz_J.npy",mat)

#mat2=numpy.load("matriz_J.npy")
