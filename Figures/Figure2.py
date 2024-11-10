
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
plt.style.use('default')
minStim = -1
maxStim = 1
stimRange = [1.1 * minStim - .1 * maxStim, 1.1 * maxStim - .1 * minStim]

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = '12'
mpl.rcParams['font.family'] = 'sans-serif'


alltrialdata = pd.read_csv('/Opto2afc/ChR2 Stimulus/alltrialdata.csv')
mice = ['Bucky','Yoyo','Ginger','Lime','Van','El']
psychfits = np.load('/Opto2afc/ChR2 Stimulus/psychfits_pChooseR_byAbsDiff.npy', allow_pickle=True)[()]
psychfits_pooled = np.load('/Opto2afc/ChR2 Stimulus/psychfits_pChooseR_byAbsDiff_POOLED.npy', allow_pickle=True)[()]

fig, ax = plt.subplots(1,5,figsize=(14,3.5))
axlist = fig.get_axes()
for k in range(5):


    for m in range(len(mice)):
        axlist[k].plot(np.linspace(stimRange[0], stimRange[1], 1000), psychfits['Fitted Curves'][m,0,0,k,:], linewidth=1, color='darkgray',alpha=0.8)
        axlist[k].plot(np.linspace(stimRange[0], stimRange[1], 1000), psychfits['Fitted Curves'][m,1,0,k,:], linewidth=1, color='#3d88c8',alpha=0.5)
    
for k in range(5):
    axlist[k].plot(np.arange(2)*0, np.arange(2), 'silver',':', linewidth=0.75,alpha=0.75)
    axlist[k].plot(np.arange(-1.5,2), np.arange(-1.5,2)*0+0.5,'silver', ':', linewidth=0.75,alpha=0.75)
     
    axlist[k].plot(np.linspace(stimRange[0], stimRange[1], 1000), psychfits_pooled['Fitted Curves'][0,0,k,:], linewidth=4, color='k')
    axlist[k].plot(np.linspace(stimRange[0], stimRange[1], 1000), psychfits_pooled['Fitted Curves'][1,0,k,:], linewidth=4, color='#3d88c8')
    
    axlist[k].set_ylim([0,1])
    axlist[k].set_xlim([-1.05,1.05])
#plt.savefig('OptobyAbsDiff.pdf')