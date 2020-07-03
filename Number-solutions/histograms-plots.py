# import matplotlib
# matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.optimize import curve_fit
import csv
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import math
import time
from scipy.stats import norm
from collections import Counter

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import collections
import numpy as np
import numpy.matlib
import time
import random
import os
import errno
import csv
from scipy.stats import norm
from astroML.density_estimation import GaussianMixture1D

# te=0.31827586
# init=time.time()
# data = pd.read_csv('total_{}.csv'.format(te))

# xx=list(data['energia'])
# yx=list(data['parametro'])

# np.save("total_x_{}.npy".format(te),xx)
# np.save("total_y_{}.npy".format(te),yx)

# x=np.load("total_x_{}.npy".format(te))
# y=np.load("total_y_{}.npy".format(te))

# x_filtr=list(set(x))
# y_filtr=list(set(y))

# print(len(y_filtr))

# data1 = pd.read_csv('1total_{}.csv'.format(te))

# xx=list(data1['interaction'])
# yx=list(data1['parametro'])
# x1=np.load("1total_x_{}.npy".format(te))
# y1=np.load("1total_y_{}.npy".format(te))

# x1_filtr=list(set(x1))
# y1_filtr=list(set(y1))

#######################################################
#######################################################
################# Histograma ed-and ###################
#######################################################
#######################################################
# mu, std = norm.fit(y)

# # Plot the histogram.
#plt.hist(y, bins=len(y_filtr), density=True, color='k')

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()
#temp=[0.369,0.06,0.780]
temp=[0.1]
for elem in temp:

    te=elem
    init=time.time()
    data = pd.read_csv('total_{}.csv'.format(te))

    xx=list(data['energia'])
    yx=list(data['parametro'])

    np.save("total_x_{}.npy".format(te),xx)
    np.save("total_y_{}.npy".format(te),yx)

    y=list(np.load("total_y_{}.npy".format(te)))
    x1=list(np.load("total_x_{}.npy".format(te)))
    print("numero de elementos antes:",len(y))
    yasum1=list(np.load("total_y_{}.npy".format(te)))
    xasum1=list(np.load("total_x_{}.npy".format(te)))
    yasum2=list(np.load("total_y_{}.npy".format(te)))
    xasum2=list(np.load("total_x_{}.npy".format(te)))
    yasum3=list(np.load("total_y_{}.npy".format(te)))
    xasum3=list(np.load("total_x_{}.npy".format(te)))
    yasum3=list(np.load("total_y_{}.npy".format(te)))
    xasum3=list(np.load("total_x_{}.npy".format(te)))
    k = 0
    for i in yasum1:
        yasum1[k] -= 0.05
        xasum1[k] -= 0.05
        yasum2[k] += 0.05
        xasum2[k] += 0.05
        yasum3[k] -= 0.02
        xasum3[k] -= 0.02
        k+=1    
    print("primeros elementos:",y[0],yasum1[0])
    y=y+yasum1+yasum2+yasum3
    x1=x1+xasum1+xasum2+xasum3
    print("numero de elementos despues:",len(y))
    y_filtr=list(set(y))
    x_filtr=list(set(x1))
    print(te,len(y_filtr))


    cnts = Counter(x1)
    maximum_cnt = max(cnts.values())
    print("maximun frequency:", maximum_cnt)

    sample_mu = np.mean(y)
    sample_std = np.std(y, ddof=1)

    fig, ax = plt.subplots(figsize=(5, 3.75))

    ax.hist(y, len(y_filtr), density=True, color='k')
    x = np.linspace(-2.1, 4.1, 1000)


    ax.plot(x, norm.pdf(x, sample_mu, sample_std), '--k', label='Gaussian fit')

    ax.legend(loc=1)
    ax.set_xlim(1, 0)
    ax.set_ylim()
    ax.set_xlabel('Interaction Energy')
    ax.set_ylabel('Counts')
    ax.text(0.95, 0.80, ('$\mu = {:.3f}$\n $\sigma={:.3f}$\n $T={}$'.format(sample_mu,sample_std,te)),
        transform=ax.transAxes, ha='right', va='top')
    plt.savefig("ED-AND-histogram-N=20-T{}.png".format(te))
    plt.show()
    plt.close()
    
    
    
    
    sample_mu = np.mean(x1)
    sample_std = np.std(x1, ddof=1)

    fig, ax = plt.subplots(figsize=(5, 3.75))

    ax.hist(x1, len(x_filtr), density=True, color='k')
    xx = np.linspace(-2.1, 4.1, 1000)


    ax.plot(xx, norm.pdf(xx, sample_mu, sample_std), '--k', label='Gaussian fit')

    ax.legend(loc=1)
    ax.set_xlim(-0.7,-0.3)

    ax.set_xlabel('$q_{EA}$')
    ax.set_ylabel('Counts')
    ax.text(0.95, 0.80, ('$\mu_1 = {:.3f}$\n $\sigma_1={:.3f}$\n $T={}$'.format(sample_mu,sample_std,te)),
        transform=ax.transAxes, ha='right', va='top')
    plt.savefig("Interaction-energy-histogram-N=20-T={}-65536.png".format(te))
    plt.show()
    plt.close()



























# HIS_q=plt.figure()
# n2, bins2, patches2 = plt.hist(y,bins=len(y_filtr),color='black')
# HIS_q.show()

# HIS_q1=plt.figure()
# n21, bins21, patches21 = plt.hist(y1,bins=len(y1_filtr),color='black')
# HIS_q1.show()


#mat2=numpy.load("matriz_J.npy")

# HIS_q=plt.figure(figsize=(10,10))
# n2, bins2, patches2 = plt.hist(y,bins=len(x_filtr),color='black')
# HIS_q.show()

# # definitions for the axes
# left, width = 0.75, 0.1
# bottom, height = 0.5, 0.1
# spacing = 0.005

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom + height + spacing, width, 0.1]
# rect_histy = [left + width + spacing, bottom, 0.1, height]

# #start with a rectangular Figure
# plt.figure(figsize=(30, 30))

# ax_scatter = plt.axes(rect_scatter)
# ax_scatter.tick_params(direction='in', top=True, right=True)
# ax_histx = plt.axes(rect_histx)
# ax_histx.tick_params(direction='in', labelbottom=False)
# ax_histy = plt.axes(rect_histy)
# ax_histy.tick_params(direction='in', labelleft=False)

# # the scatter plot:
# ax_scatter.scatter(x, y,color='black')

# ax_scatter.set_xlim((-0.8, 0))
# #ax_scatter.set_ylim((-1,1))
# ax_scatter.set_ylim((0.85,1))
# ax_histx.hist(x, bins=len(x_filtr),color='black')
# ax_histy.hist(y, bins=len(y_filtr), orientation='horizontal',color='black')
# #ax_histx.set_xlim(-0.8,0)
# #ax_histy.set_ylim(0.75,1)

# ax_histx.set_xlim(ax_scatter.get_xlim())
# ax_histy.set_ylim(ax_scatter.get_ylim())

# ax_histx.set_ylabel('counts')
# ax_histy.set_xlabel('counts')
# ax_scatter.set_xlabel('Energy')
# ax_scatter.set_ylabel('Ed-Ad')
# plt.show()


# def myplot(x, y, s, bins=500):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=s)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# #    extent = [-0.8,-0.3,0.7,1]
#     print(xedges[0], xedges[-1], yedges[0], yedges[-1])
#     return heatmap.T, extent


# plt.show()
# img, extent = myplot(x, y, 16)
# plt.figure(figsize=(10, 10))
# #plt.plot(x, y, 'k.', markersize=5)
# plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
# plt.title("Smoothing with  $\sigma$ = %d" % 64)
# plt.colorbar(orientation='horizontal',fraction=.1)











# grid_size=0.001
# h=0.1

# x_min=min(x_filtr)
# x_max=max(x_filtr)
# y_min=min(y_filtr)
# y_max=max(y_filtr)

# x_grid=np.arange(x_min-h,x_max+h,grid_size)
# y_grid=np.arange(y_min-h,1.01,grid_size)
# x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

# xc=x_mesh+(grid_size/2)
# yc=y_mesh+(grid_size/2)

# def kde_quartic(d,h):
#     dn=d/h
#     P=(15/16)*(1-dn**2)**2
#     return P

# intensity_list=[]
# for j in range(len(xc)):
#     intensity_row=[]
#     for k in range(len(xc[0])):
#         kde_value_list=[]
#         for i in range(len(x)):
#             #CALCULATE DISTANCE
#             d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
#             if d<=h:
#                 p=kde_quartic(d,h)
#             else:
#                 p=0
#             kde_value_list.append(p)
#         #SUM ALL INTENSITY VALUE
#         p_total=sum(kde_value_list)
#         intensity_row.append(p_total)
#     intensity_list.append(intensity_row)
    
# intensity=np.array(intensity_list)
# FIG=plt.figure()
# plt.pcolormesh(x_mesh,y_mesh,intensity)
# plt.plot(x,y,'ro',markersize=2)
# plt.colorbar()
# plt.show()
# FIG.savefig('mapa_calor.png')
# plt.close()
# fini=time.time()
# print(fini-init)