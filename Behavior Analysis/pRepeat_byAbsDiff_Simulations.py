
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')


alltrialdata = pd.read_csv('/Users/ningleow/Library/CloudStorage/GoogleDrive-ningleow@mit.edu/My Drive/Full 2AFC with No Opto/alltrialdata.csv')
mice = np.unique(alltrialdata['Mouse'])

alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
allsimulated = pd.DataFrame()
allsimulatedwBias = pd.DataFrame()

for m in range(len(mice)):
    trialdata = alltrialdata[alltrialdata.Mouse==mice[m]]
    chooseRrate = dict(trialdata.groupby('DirCoh')['ChooseR'].mean())
    prevR_Rrate = dict(trialdata[trialdata['Previous Choice']=='R'].groupby('DirCoh')['ChooseR'].mean()) # get chooseR after prevR bias
    prevL_Rrate = dict(trialdata[trialdata['Previous Choice']=='L'].groupby('DirCoh')['ChooseR'].mean()) # get chooseR after prevL bias

    for i in range(100):
        #### WITH 0 BIAS
        simulated = pd.DataFrame()
        simulated['AbsDiff'] = trialdata['AbsDiff']
        simulated['DirCoh'] = trialdata['DirCoh']

        hold = []
        for tr in range(len(trialdata)):
            # Binomial sampling
            hold.append(np.random.binomial(1, chooseRrate[simulated['DirCoh'].iloc[tr]], 1))

        simulated['ChooseR'] = hold
        simulated['Mouse'] = mice[m]
        simulated['Previous Outcome'] = trialdata['Previous Outcome']
        simulated['Previous ChooseR'] = np.nan
        simulated['Previous ChooseR'].iloc[1:] = simulated['ChooseR'].iloc[:-1].to_numpy()
        simulated['Repeat'] = simulated['Previous ChooseR']==simulated['ChooseR']

        allsimulated = allsimulated.append(simulated, ignore_index=True)

        #### WITH HISTORY BIAS
        simulated = pd.DataFrame()
        simulated['AbsDiff'] = trialdata['AbsDiff']
        simulated['DirCoh'] = trialdata['DirCoh']

        hold = [0]
        for tr in range(1,len(trialdata)):
            # Binomial sampling with bias
            if hold[-1] == 1:
                hold.append(np.random.binomial(1, prevR_Rrate[simulated['DirCoh'].iloc[tr]], 1))
            elif hold[-1] == 0:
                hold.append(np.random.binomial(1, prevL_Rrate[simulated['DirCoh'].iloc[tr]], 1))

        simulated['ChooseR'] = hold
        simulated['Mouse'] = mice[m]
        simulated['Previous Outcome'] = trialdata['Previous Outcome']
        simulated['Previous ChooseR'] = np.nan
        simulated['Previous ChooseR'].iloc[1:] = simulated['ChooseR'].iloc[:-1].to_numpy()
        simulated['Repeat'] = simulated['Previous ChooseR']==simulated['ChooseR']

        allsimulatedwBias = allsimulatedwBias.append(simulated, ignore_index=True)


### Obtain slopes
slopesdata = []
slopessimulated = []
slopessimulatedwbias = []

for m in range(len(mice)):
    slopesdata.append(np.polyfit(np.unique(alltrialdata[alltrialdata.Mouse==mice[m]]['AbsDiff']),alltrialdata[alltrialdata.Mouse==mice[m]].pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(),1)[0])
    slopessimulated.append(np.polyfit(np.unique(allsimulated[allsimulated.Mouse==mice[m]]['AbsDiff']),allsimulated[allsimulated.Mouse==mice[m]].pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(),1)[0])
    slopessimulatedwbias.append(np.polyfit(np.unique(allsimulatedwBias[allsimulatedwBias.Mouse==mice[m]]['AbsDiff']),allsimulatedwBias[allsimulatedwBias.Mouse==mice[m]].pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(),1)[0])



### Plot pRep by AbsDiff
plt.figure(figsize=(3,4))

plt.scatter(np.unique(allsimulated['AbsDiff']),allsimulated.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(), color='w',edgecolor='silver')
vals = np.unique(allsimulated['AbsDiff'])
m, c = np.polyfit(np.unique(allsimulated['AbsDiff']),np.nanmean(allsimulated.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat'),0),1)
y=m*vals + c
plt.plot(vals,y, color='silver')

plt.scatter(np.unique(allsimulatedwBias['AbsDiff']),allsimulatedwBias.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(),color='w', edgecolor='gray')
vals = np.unique(allsimulatedwBias['AbsDiff'])
m, c = np.polyfit(np.unique(allsimulatedwBias['AbsDiff']),np.nanmean(allsimulatedwBias.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat'),0),1)
y=m*vals + c
plt.plot(vals,y, color='gray')#, color='silver')

vals = np.unique(alltrialdata['AbsDiff'])
m, c = np.polyfit(np.unique(alltrialdata['AbsDiff']),np.nanmean(alltrialdata.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat'),0),1)
y=m*vals + c
plt.scatter(np.unique(alltrialdata['AbsDiff']),alltrialdata.pivot_table(index='Mouse',columns='AbsDiff',values='Repeat').mean(), color='#3C5488')
plt.plot(vals,y, color='#3C5488')

plt.ylim([0.2,0.8])

