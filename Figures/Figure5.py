import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pingouin as pg
plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42


### Load Data
rootdata = '/Users/ningleow/Library/CloudStorage/GoogleDrive-ningleow@mit.edu/My Drive/rdk-2afc/CDAnalysisPooled/'
projs = np.load(rootdata + 'proj_stim.npy', allow_pickle=True) # iterations x cd x taskfactor x time x prevoutcome x proj
projs_absdiff = np.load(rootdata + 'proj_stim_byabsdiff.npy', allow_pickle=True) # iterations x cd x taskfactor x time x absdiffbin x prevoutcome x proj

### Set colors
cohcolors = ['#313131','#5d5d51','#8c8c73','#bfbf97']
dircohcolors = ['#003c30','#01665e','#35978f','#80cdc1','#dfc27d','#bf812d','#8c510a','#543005']
from palettable.scientific.sequential import LaPaz_6_r



#%%# Fig 5b,c

fig, ax = plt.subplots(1,2, figsize=(4,3), sharey='row')
axlist = fig.get_axes()

# By dir x coh
for i in range(8):
    array = np.nanmean(projs[:,0,0,:,0,i],0)
    axlist[0].plot(array, c=dircohcolors[i], linewidth=3) 

for i in range(4):
    array = np.nanmean(projs[:,3,1,:,0,i],0)
    axlist[1].plot(array, c=cohcolors[i], linewidth=3) 

for i in range(len(axlist)):
    axlist[i].spines['right'].set_visible(False)
    axlist[i].spines['top'].set_visible(False)
    axlist[i].axvspan(xmin=20,xmax=35, facecolor='silver',alpha=0.5)
    axlist[i].plot(np.arange(60),np.arange(60)*0,'k--')
    #axlist[i].axvspan(xmin=34.5,xmax=35.5, facecolor='silver',alpha=1)

    axlist[i].set_xlim([15,40])


#%%# Fig 5d

fig, ax = plt.subplots(1,8, figsize=(10,3), sharey='row')
axlist = fig.get_axes()

for x in range(5): # Per |∆Dir| Bin
    for i in range(8): # Per stimulus
        means = np.nanmean(np.nanmean(projs_absdiff[:,0,0,:,x,0,i],-1),0)#np.nanmean(np.nanmean(projs[:,0,0,:,:,0,:],-1),-1)
        array = np.nanmean(projs_absdiff[:,0,0,:,x,0,i],0)#-means,0)

        axlist[i].plot(array, c=LaPaz_6_r.mpl_colors[x+1], linewidth=3) #axlist[i].plot(np.nanmean(array, 0), c=Broc_7.mpl_colors[x], linewidth=3)


for i in range(len(axlist)):
    axlist[i].spines['right'].set_visible(False)
    axlist[i].spines['top'].set_visible(False)
    axlist[i].axvspan(xmin=20,xmax=35, facecolor='silver',alpha=0.5)
    axlist[i].plot(np.arange(60),np.arange(60)*0,'k--')

    axlist[i].set_xlim([15,40])
    
#plt.savefig('traject.pdf')

#%%# Fig 5e

points = np.zeros([100, 5, 8]) # iterations x absdiffbin x dircoh
slopes = np.zeros([100, 8]) # iterations x dircoh

# Take the mean projection of the end of the stimulus period
for i in range(100):

    for x in range(5):
        for k in range(8):
            array = projs_absdiff[i,0,0,:,x,0,k] 
            point = np.nanmean(array[25:35])

            points[i,x,k] = point

for i in range(100):
    for k in range(8):
    
        holdpoints = points[i,:,k]
        holdpoints=holdpoints[~np.isnan(holdpoints)]

        # Linear fit of projection points to |∆Dir| bins
        slopes[i,k]=np.polyfit(np.arange(len(holdpoints)), holdpoints, 1)[0]

plt.figure(figsize=(3,4))
plt.errorbar(np.arange(8), np.nanmean(slopes,0), yerr=stats.sem(slopes,0, nan_policy='omit'), capsize=3, fmt='-o')
plt.xlim([-1,8])


#%%# Fig 5f
# Plot the trajectories along DirectionCD and CoherenceCD for each bin of |∆Dir|

nlevels = 5 # 5 |∆Dir| bins
fig, ax = plt.subplots(1,nlevels, figsize=(14,3), sharey=True, sharex=True)
axlist = fig.get_axes()

diraxis = np.zeros([nlevels, 8, 15])*np.nan
cohaxis = np.zeros([nlevels, 8, 15])*np.nan

for x in range(nlevels):
    for k in range(8):

        diraxis[x,k] = np.nanmean(projs_absdiff[:,0,0,20:35,x,0,k],0) 
        cohaxis[x,k] = np.nanmean(projs_absdiff[:,0,1,20:35,x,0,k],0) 
for x in range(nlevels):
    for k in range(8):
        # Plot from same origin
        axlist[x].plot(diraxis[x,k]-diraxis[x,k][0], cohaxis[x,k]-cohaxis[x,k][0], c=dircohcolors[k], linewidth=2)
        axlist[x].scatter(diraxis[x,k,-1]-diraxis[x,k][0], cohaxis[x,k,-1]-cohaxis[x,k][0], c=dircohcolors[k], linewidth=2,edgecolors='k', s=150)
        axlist[x].scatter(diraxis[x,k,0]-diraxis[x,k][0], cohaxis[x,k,0]-cohaxis[x,k][0], c='silver', linewidth=2,edgecolors='silver', s=20)


#%%# Fig 5g
# Euclidean distance between stimuli at each coherence level along DirectionCD and CoherenceCD 

LRdiff = np.zeros([100,4,5])*np.nan

for ad in range(nlevels):
    for coh in range(4):
        for j in range(100):

            LRdiff[j,coh,ad] = math.dist([np.nanmean(projs_absdiff[j,0,0,30:35,ad,0,coh]), np.nanmean(projs_absdiff[j,0,1,30:35,ad,0,coh])], [np.nanmean(projs_absdiff[j,0,0,30:35,ad,0,7-coh]), np.nanmean(projs_absdiff[j,0,1,30:35,ad,0,7-coh])])

plt.plot(np.nanmean(np.nanmean(LRdiff,1),0))

# ANOVA

diffhold = pd.DataFrame(np.nanmean(LRdiff,1), columns=[0,1,2,3,4])
diffhold['Iteration'] = np.arange(100)
diffmelt = pd.melt(diffhold, id_vars=['Iteration'], var_name='AbsDiff', value_name='value')
pg.rm_anova(data=diffmelt, dv='value', subject='Iteration', within='AbsDiff', effsize='np2')
#%%# Fig 5h

distbetw = np.zeros([100,8,5])*np.nan
import math

for x in range(nlevels):
    for k in range(8):
        for i in range(100):

            # Distance from the start of the stimulus to mean at end of stimulus, for dir axis and coh axis
            distbetw[i,k,x] = math.dist([np.nanmean(projs_absdiff[i,0,0,19,x,0,k]), np.nanmean(projs_absdiff[i,0,1,19,x,0,k])], [np.nanmean(projs_absdiff[i,0,0,30:35,x,0,k]), np.nanmean(projs_absdiff[i,0,1,30:35,x,0,k])])


# Format data for ANOVA
distL = pd.DataFrame(np.nanmean(distbetw[:,:4,:],1), columns=[0,1,2,3,4])
distL['Iteration'] = np.arange(100)
distLmelt = pd.melt(distL, id_vars=['Iteration'], var_name='AbsDiff', value_name='dist')
distLmelt['Direction'] = 'L'

distR = pd.DataFrame(np.nanmean(distbetw[:,4:,:],1), columns=[0,1,2,3,4])
distR['Iteration'] = np.arange(100)
distRmelt = pd.melt(distR, id_vars=['Iteration'], var_name='AbsDiff', value_name='dist')
distRmelt['Direction'] = 'R'

distjoint = pd.concat([distLmelt,distRmelt])
pg.rm_anova(data=distjoint, dv='dist', subject='Iteration', within=['AbsDiff','Direction'], effsize='np2')