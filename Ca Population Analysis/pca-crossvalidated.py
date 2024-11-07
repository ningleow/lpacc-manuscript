
#################################
#### Performs cross-validated PCA based on trial averaged data,
#### Plots PCA comparison between task-engaged and passive conditions
#### Computes pair-wise Euclidean distances across 8 stimuli
#################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
import math
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
rootdata = '/Users/ningleow/Library/CloudStorage/GoogleDrive-ningleow@mit.edu/My Drive/rdk-2afc/2P/'
sessions = [x for x in next(os.walk(rootdata))[1]]

sessions = np.array(sessions)[np.argsort(sessions)]

####


alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
allcoh = [1,0.64,0.32,0.16]
nbigiters = 100 # no. of splits, multiple iterations to perform euclidean distance comparison
nIters = 100 # trial based bootstrap iterations
alltestprojs = np.zeros([nbigiters, 10, 8, 40])*np.nan
expvar = np.zeros([nIters, 10])*np.nan

euclidean = np.zeros([nbigiters, 8, 8])*np.nan

for j in range(nbigiters):
    for sesh in range(len(sessions)):

        session = sessions[sesh]
        print(session)
        path = rootdata + session
        fulldata = np.load(path + '/fulldata.npy', allow_pickle=True)[()]
        dFF = fulldata['dFF']
        nUnits = np.shape(dFF)[0]
        starts = fulldata['roi_start'].astype('int')
        ends = fulldata['roi_end'].astype('int')
        trialdata = fulldata['trialdata'].iloc[:-1, :]
        trialdata = trialdata[(trialdata.Outcomes == 'Hit')|(trialdata.Outcomes == 'FA')]

        nUnits = np.shape(dFF)[0]

        dFF = dFF[:, :int(np.percentile(ends, 99))]
        trialdata = trialdata[trialdata.StimStart < int(np.percentile(ends, 90))].iloc[:-1]
        nTrials = len(trialdata)
        for n in range(nUnits):
            dFF[n, :starts[n]] = np.nan
            dFF[n, ends[n]:] = np.nan

        dFF = pd.DataFrame(dFF).dropna(axis=0).to_numpy()
        nUnits = np.shape(dFF)[0]

        normdFF = StandardScaler().fit_transform(dFF.transpose())

        triallen = 40
        dFFbytrial_stim = np.zeros([nUnits, (nTrials),triallen])*np.nan

        for tr in range(nTrials):

            dFFbytrial_stim[:,tr,:] = normdFF[:,int(trialdata.iloc[tr]['StimStart'])-20:int(trialdata.iloc[tr]['StimStart'])+20] 

        
        ### Split trials into 2 equal halves, stratified by DirxCoh for balance
        train_idx, test_idx = train_test_split(np.arange(nTrials), test_size=0.5, stratify=trialdata['DirCoh'])
        
        trainset = np.zeros([nUnits, nIters, 8, triallen])*np.nan
        testset = np.zeros([nUnits, nIters, 8, triallen])*np.nan


        # Bootstrapped trial means, choose 5 trials each time
        for i in range(nIters):
            for dc in range(8):

                sub = np.random.choice(np.where(trialdata.iloc[train_idx]['DirCoh']==alldircoh[dc])[0], 5, replace=True)
                trainset[:,i,dc,:] = np.nanmean(dFFbytrial_stim[:,train_idx[sub],:],1)

                sub = np.random.choice(np.where(trialdata.iloc[test_idx]['DirCoh']==alldircoh[dc])[0], 5, replace=True)
                testset[:,i,dc,:] = np.nanmean(dFFbytrial_stim[:,test_idx[sub],:],1)

        if sesh == 0:
            alltrain = trainset
            alltest = testset
        else:
            alltrain = np.vstack((alltrain, trainset))
            alltest = np.vstack((alltest, testset))



    nUnits = np.shape(alltrain)[0]
    meantrain = np.mean(alltrain,1)
    meantest = np.mean(alltest,1)

    trainpca = np.reshape(meantrain[:,:,23:33], (nUnits, int(8*10)))
    pca = PCA(n_components=10, svd_solver='full', whiten=True)
    fit = pca.fit(trainpca.transpose())

    # PCs, dirxcoh, time
    trainproj =  np.zeros([10, 8, triallen])*np.nan 
    testproj =   np.zeros([10, 8, triallen])*np.nan

    for dc in range(8):

        trainproj[:,dc,:] = pca.transform(alltrain[:,dc,:].transpose()).transpose()
        testproj[:,dc,:] = pca.transform(meantest[:,dc,:].transpose()).transpose()

    alltestprojs[j] = testproj
    expvar[j] = np.cumsum(pca.explained_variance_ratio_)


    for dc1 in range(8):
        for dc2 in range(8):
            euclidean[j,dc1,dc2] = math.dist(np.nanmean(testproj[:3,dc1,23:30],-1), np.nanmean(testproj[:3,dc2,23:30],-1))

####
# Passive imaging PCA
loadpassive = np.load('passivepcadata.npy',allow_pickle=True)[()]
plt.figure(figsize=(4,5))
plt.errorbar(np.arange(10), np.nanmean(expvar,0), yerr=stats.sem(expvar,0,nan_policy='omit'),fmt='o-', capsize=3,color='#5d98c9')
plt.errorbar(np.arange(10), np.nanmean(loadpassive['Explained Variance'],0), yerr=np.std(loadpassive['Explained Variance'],0),fmt='o-', capsize=3, color='gray')

plt.ylim([0,1])
plt.xlim([-1,10])

#### 
# Visualize first few PCs
dircohcolors = ['#003c30','#01665e','#35978f','#80cdc1','#dfc27d','#bf812d','#8c510a','#543005']
cohcolors = ['#313131','#5d5d51','#8c8c73','#bfbf97']
plt.style.use('default')
nPCs = 5
fig, ax = plt.subplots(2,nPCs, figsize=(12,4.5), sharey='col')
axlist = fig.get_axes()

testproj_coh = np.zeros([20, 4, 60])*np.nan
testproj_coh[:,0,:] = np.nanmean(testproj[:,[0,7],:],1)
testproj_coh[:,1,:] = np.nanmean(testproj[:,[1,6],:],1)
testproj_coh[:,2,:] = np.nanmean(testproj[:,[2,5],:],1)
testproj_coh[:,3,:] = np.nanmean(testproj[:,[3,4],:],1)


for x in range(nPCs):
    for dc in range(8):
        axlist[x].plot(testproj[x,dc,:], c=dircohcolors[dc], linewidth=3)
    for k in range(4):
        axlist[x+nPCs].plot(testproj_coh[x,k,:], c=cohcolors[k], linewidth=3)

for x in range(len(axlist)):
    axlist[x].set_xlim([10,50])
    axlist[x].axvspan(xmin=20,xmax=35, facecolor='silver',alpha=0.4)
    axlist[x].spines['top'].set_visible(False)
    axlist[x].spines['right'].set_visible(False)

plt.tight_layout()

#plt.savefig('4PCs.pdf')

########
### Plot 3D
########

t=30
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
for dc in range(8):
    ax.scatter(testproj[2,dc,t],testproj[0,dc,t],testproj[1,dc,t], c=dircohcolors[dc], s=100, edgecolor='k')
ax.set_xlabel('PC0')
ax.set_ylabel('PC2')
ax.set_zlabel('PC1')
ax.view_init(elev=15,  azim=53)
