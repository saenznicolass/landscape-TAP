import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.optimize import curve_fit
import csv

data = pd.read_csv('j3/qea_vs_energy_aveplane_303.csv')

x=list(data['energia'])
y=list(data['parametro'])
x_filtr=list(set(x))
print('N:19',"total:", len(x), 'filtrado:', len(x_filtr))


HIS_q=plt.figure()
n2, bins2, patches2 = plt.hist(y,bins=(100),color='black')
plt.xlabel('q_EA')
plt.ylabel('counts')
# plt.ylabel('Counts')
plt.xlim(0.6,1)
#plt.subplots_adjust(left=0.15)
HIS_q.savefig("qea-hist.png")

# definitions for the axes
# left, width = 0.75, 0.1
# bottom, height = 0.5, 0.1
# spacing = 0.005

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom + height + spacing, width, 0.1]
# rect_histy = [left + width + spacing, bottom, 0.1, height]

# # start with a rectangular Figure
# plt.figure(figsize=(30, 30))

# ax_scatter = plt.axes(rect_scatter)
# ax_scatter.tick_params(direction='in', top=True, right=True)
# ax_histx = plt.axes(rect_histx)
# ax_histx.tick_params(direction='in', labelbottom=False)
# ax_histy = plt.axes(rect_histy)
# ax_histy.tick_params(direction='in', labelleft=False)

# # the scatter plot:
# ax_scatter.scatter(x, y)

# # now determine nice limits by hand:
# binwidth = 0.01
# lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
# ax_scatter.set_xlim((-0.8, 0))
# #ax_scatter.set_ylim((-1,1))
# ax_scatter.set_ylim((0.85,1))

# bins = np.arange(-1, 1 + binwidth, binwidth)
# print(bins)
# ax_histx.hist(x, bins=bins)
# ax_histy.hist(y, bins=bins*1, orientation='horizontal')
# #ax_histx.set_xlim(-0.8,0)
# #ax_histy.set_ylim(0.75,1)

# ax_histx.set_xlim(ax_scatter.get_xlim())
# ax_histy.set_ylim(ax_scatter.get_ylim())

# ax_histx.set_ylabel('counts')
# ax_histy.set_xlabel('counts')
# ax_scatter.set_xlabel('Energy')
# ax_scatter.set_ylabel('Ed-Ad')
# plt.show()