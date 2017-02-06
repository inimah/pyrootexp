import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def readData(filename):
    
    #headerColumns = ['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z','mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'dr','dphir','dz']
    headerColumns = ['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z','mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'class']
    dataframe = pd.read_csv(filename, header=None, names = headerColumns)

    return dataframe


#filename = "/home/tita/PyRoot/data/data171618_Reg_a.txt"
filename = "/home/tita/PyRoot/data/data171618_Class_a.txt"
dataframe = readData(filename)
dataset = dataframe.values
X = dataset[:,0:33]
y = dataset[:,-1]

x_input = X[:,:3]
fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, elev=-150, azim=110)
cax = ax.scatter(x_input[:, 0], x_input[:, 1], x_input[:, 2], c=y,cmap=plt.cm.Paired)

# index i = r
# index j = z
# index m = phi
ax.set_title("Cylindrical coordinates (r, z, phi)")
ax.set_xlabel("r (index i)")
ax.set_xlim3d(0, 15)
ax.w_xaxis.set_ticklabels(['0', '5', '10', '15'])
ax.set_ylabel("z (index j)")
ax.set_ylim3d(0, 15)
ax.w_yaxis.set_ticklabels(['0', '5', '10', '15'])
ax.set_zlabel("phi (index m)")
ax.set_zlim3d(0, 15)
ax.w_zaxis.set_ticklabels(['0', '5', '10', '15'])
cbar = fig.colorbar(cax, ticks=[0, 20, 69, 139, 243], shrink=0.75)
fig.patch.set_alpha(0.0)
#cbar.ax.set_yticklabels(['0', '44', '143', '461', '611'])

#plt.show()
plt.savefig('/home/tita/PyRoot/plots/3Dplot.png')

dat10p = dat[(dat['drClass']<=10)] 
dat10n = dat[(dat['drClass']>20) & (dat['drClass']<=30)]
dat10 = dat10p.append(dat10n, ignore_index=True)






