
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
plt.matplotlib.style.use('default')
from sklearn.model_selection import train_test_split

def flipaxis(axis, projA, projB, projmean, windowstart, windowend):

    import numpy as np
    A = np.nanmean(projA[axis,windowstart:windowend],-1) - projmean[axis]
    B = np.nanmean(projB[axis,windowstart:windowend],-1) - projmean[axis]

    if np.logical_and((A > 0),(B < 0)):
        flip = False
    elif np.logical_and((A < 0),(B > 0)):
        flip = True
    else:
        flip = False
    return flip


def selectivityaxis(varA, varB):
    import numpy as np
    axis = (np.nanmean(varA,-1) - np.nanmean(varB,-1)) / np.sqrt(np.nanvar((varA),-1)+np.nanvar((varB),-1))
    axis = axis / np.linalg.norm(axis, 1)

    return axis

def bootstrappedmeans(dFFbytrial, probevar, probeval, trialdataslice, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]

    full = np.zeros([nUnits, triallen, nIters])

    cohs = [1, 0.64, 0.32, 0.16]
    outcomes = ['Hit', 'FA', 'Omission']
    choices = ['R','L']

    for i in range(nIters):
        if probevar == 'CorrectSide':
            hold2 = np.zeros([nUnits, triallen, 7])*np.nan

            for ch in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan
            for otc in range(3):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Outcomes==outcomes[otc]))[0]
                    hold2[:,:,4+otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,4+otc] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'DotsCoh':
            hold2 = np.zeros([nUnits, triallen, 5])*np.nan
            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Choice==choices[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan
            for otc in range(3):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Outcomes==outcomes[otc]))[0]
                    hold2[:,:,2+otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,2+otc] = np.nan
            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Outcomes':
            hold2 = np.zeros([nUnits, triallen, 6])*np.nan

            for ch in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan

            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Choice==choices[ch]))[0]
                    hold2[:,:,4+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,4+ch] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Choice':
            hold2 = np.zeros([nUnits, triallen, 8])*np.nan

            for ch in range(4):
                for otc in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]) & (trialdataslice.Outcomes==outcomes[otc]))[0]
                        hold2[:,:,otc*2+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,otc*2+ch] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Previous Correct Side':
            hold2 = np.zeros([nUnits, triallen, 7])*np.nan

            for ch in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]) & (trialdataslice.Outcomes=='Hit'))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan


            full[:,:,i] = np.nanmean(hold2, -1)


        if probevar == 'Previous Coh':
            hold2 = np.zeros([nUnits, triallen, 5])*np.nan
            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Choice==choices[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan
            for otc in range(3):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Outcomes==outcomes[otc]))[0]
                    hold2[:,:,2+otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,2+otc] = np.nan
            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Previous Outcome':
            hold2 = np.zeros([nUnits, triallen, 6])*np.nan

            for ch in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan

            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.Choice==choices[ch]))[0]
                    hold2[:,:,4+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,4+ch] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Previous Choice':
            hold2 = np.zeros([nUnits, triallen, 4])*np.nan

            for ch in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]) & (trialdataslice.Outcomes=='Hit'))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan



            full[:,:,i] = np.nanmean(hold2, -1)


        if probevar == 'AbsDiffBin':
            hold2 = np.zeros([nUnits, triallen, 2])*np.nan

            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.CorrectSide==choices[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)


        if probevar == 'DiffBin':
            hold2 = np.zeros([nUnits, triallen, 2])*np.nan

            for ch in range(2):
                try:
                    IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.CorrectSide==choices[ch]))[0]
                    hold2[:,:,ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,ch] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Omitted':
            hold2 = np.zeros([nUnits, triallen, 8])*np.nan

            for ch in range(4):
                for q in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==probeval)& (trialdataslice.DotsCoh==cohs[ch]) & (trialdataslice.CorrectSide==choices[q]))[0]
                        hold2[:,:,ch*2+q] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,ch*2+q] = np.nan

            full[:,:,i] = np.nanmean(hold2, -1)

    return np.nanmean(full,-1)

def bootstrapped_projset(dFFbytrial, probevar, trialdataslice, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]


    alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
    cohs = [1, 0.64, 0.32, 0.16]
    outcomes = ['Hit', 'FA']
    choices = ['R','L']

    full = np.zeros([nUnits, triallen, 8, nIters])
    for i in range(nIters):
        if probevar == 'DirCoh':

            hold2 = np.zeros([nUnits, triallen, 8, 3])*np.nan

            for x in range(8):
                for otc in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==alldircoh[x])& (trialdataslice.Outcomes==outcomes[otc]))[0]
                        hold2[:,:,x,otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,otc] = np.nan
            full[:,:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'DotsCoh':

            hold2 = np.zeros([nUnits, triallen, 8, 5])*np.nan

            for x in range(4):
                for ch in range(2):

                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice.CorrectSide==choices[ch]))[0]
                        hold2[:,:,x, ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,ch] = np.nan
                for otc in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice.Outcomes==outcomes[otc]))[0]
                        hold2[:,:,x,2+otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,2+otc] = np.nan

            full[:,:,:,i] = np.nanmean(hold2, -1)

        if probevar == 'Previous Coh':

            hold2 = np.zeros([nUnits, triallen, 8, 5])*np.nan

            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice['Previous Correct Side']==choices[ch]))[0]
                        hold2[:,:,x, ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,ch] = np.nan
                for otc in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice['Previous Outcome']==outcomes[otc]))[0]
                        hold2[:,:,x,2+otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,2+otc] = np.nan

            full[:,:,:,i] = np.nanmean(hold2, -1)



        if probevar == 'Outcomes':


            hold2 = np.zeros([nUnits, triallen, 8, 4])*np.nan

            for x in range(4):
                for q in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]=='Hit')& (trialdataslice.DotsCoh==cohs[x]) & (trialdataslice.CorrectSide==choices[q]))[0]
                        hold2[:,:,x,q] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,q] = np.nan
                    try:
                        IDs = np.where((trialdataslice[probevar]=='FA')& (trialdataslice.DotsCoh==cohs[x]) & (trialdataslice.CorrectSide==choices[q]))[0]
                        hold2[:,:,4+x,q] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,4+x,q] = np.nan

                for q in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]=='Hit')& (trialdataslice.DotsCoh==cohs[x]) & (trialdataslice.Choice==choices[q]))[0]
                        hold2[:,:,x,2+q] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,2+q] = np.nan
                    try:
                        IDs = np.where((trialdataslice[probevar]=='FA')& (trialdataslice.DotsCoh==cohs[x]) & (trialdataslice.Choice==choices[q]))[0]
                        hold2[:,:,4+x,2+q] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,3,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,4+x,2+q] = np.nan


            full[:,:,:,i] = np.nanmean(hold2, -1)


        if probevar == 'Previous Outcome':

            hold2 = np.zeros([nUnits, triallen, 8, 7])*np.nan

            for x in range(4):

                for otc in range(2):
                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x]) & (trialdataslice.Outcomes==outcomes[otc]) & (trialdataslice['Previous Outcome']=='Hit'))[0]
                        hold2[:,:,x,otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,otc] = np.nan

                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x])  &  (trialdataslice.Outcomes==outcomes[otc]) & (trialdataslice['Previous Outcome']=='FA'))[0]
                        hold2[:,:,4+x,otc] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,4+x,otc] = np.nan

                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x]) & (trialdataslice.CorrectSide==choices[ch]) & (trialdataslice['Previous Outcome']=='Hit'))[0]
                        hold2[:,:,x,3+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,3+ch] = np.nan

                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x])  &  (trialdataslice.CorrectSide==choices[ch]) & (trialdataslice['Previous Outcome']=='FA'))[0]
                        hold2[:,:,4+x,3+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,4+x,3+ch] = np.nan

                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x]) & (trialdataslice.Choice==choices[ch]) & (trialdataslice['Previous Outcome']=='Hit'))[0]
                        hold2[:,:,x,5+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,x,5+ch] = np.nan

                    try:
                        IDs = np.where((trialdataslice['Previous Coh']==cohs[x])  &  (trialdataslice.Choice==choices[ch]) & (trialdataslice['Previous Outcome']=='FA'))[0]
                        hold2[:,:,4+x,5+ch] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                    except ValueError:
                        hold2[:,:,4+x,5+ch] = np.nan

            full[:,:,:,i] = np.nanmean(hold2,-1)

        if probevar == 'Choice':

            hold2 = np.zeros([nUnits, triallen, 8])*np.nan

            for x in range(4):

                try:
                    IDs = np.where((trialdataslice[probevar]=='R')& (trialdataslice.DotsCoh==cohs[x]))[0]
                    hold2[:,:,x] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,x] = np.nan
                try:
                    IDs = np.where((trialdataslice[probevar]=='L')& (trialdataslice.DotsCoh==cohs[x]))[0]
                    hold2[:,:,4+x] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,4+x] = np.nan


            full[:,:,:,i] = hold2

        if probevar == 'Previous Choice':

            hold2 = np.zeros([nUnits, triallen, 8])*np.nan

            for x in range(4):
                try:
                    IDs = np.where((trialdataslice[probevar]=='R')& (trialdataslice['Previous Coh']==cohs[x]))[0]
                    hold2[:,:,x] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,x] = np.nan
                try:
                    IDs = np.where((trialdataslice[probevar]=='L')& (trialdataslice['Previous Coh']==cohs[x]))[0]
                    hold2[:,:,4+x] = np.nanmean(dFFbytrial[:,np.random.choice(IDs,5,replace=True),:],1)
                except ValueError:
                    hold2[:,:,4+x] = np.nan

            full[:,:,:,i] = hold2

    return np.nanmean(full, -1)

def bootstrapped_projset_cond(dFFbytrial, probevar, cond, trialdataslice, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]


    alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
    cohs = [1, 0.64, 0.32, 0.16]
    outcomes = ['Hit', 'FA']
    choices = ['R','L']


    full = np.zeros([nUnits, triallen, 2, 8])*np.nan

    #trialdataslice[probevar] = np.random.permutation(trialdataslice[probevar])

    if probevar == 'DirCoh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        for q in range(2):

            for x in range(8):
                try:
                    IDs = np.where((trialdataslice[probevar]==alldircoh[x]) & (trialdataslice[cond]==outcomes[q]))[0]

                    for i in range(nIters):
                        hold2[:,:,q,x,i] = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,:]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'DotsCoh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan


        for q in range(2):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x]) & (trialdataslice[cond]==outcomes[q]) & (trialdataslice.CorrectSide==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Outcomes':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan

        expanded = (trialdataslice['Outcomes']=='Hit')*trialdataslice['DotsCoh'] - (trialdataslice['Outcomes']=='FA')*trialdataslice['DotsCoh']

        for q in range(2):
            for x in range(8):
                for ch in range(2):
                    try:
                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==outcomes[q]) & (trialdataslice.CorrectSide==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Choice':
        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        expanded =(trialdataslice['Choice']=='R')*trialdataslice['DotsCoh'] - (trialdataslice['Choice']=='L')*trialdataslice['DotsCoh']
        for q in range(2):
            for x in range(8):

                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==outcomes[q]))[0]# & (trialdataslice.CorrectSide==choices[ch]))[0]
                    for i in range(nIters):
                        hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,i]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'Previous Coh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan


        for q in range(2):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice[cond]==outcomes[q]) & (trialdataslice['Previous Correct Side']==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,ch,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Outcome':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 8, nIters])*np.nan

        expanded = (trialdataslice['Previous Outcome']=='Hit')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Outcome']=='FA')*trialdataslice['Previous Coh']

        for q in range(2):
            for x in range(8):
                for pdc in range(8):
                    try:
                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==outcomes[q]) & (trialdataslice['Previous DirCoh']==alldircoh[pdc]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,pdc,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,pdc,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Choice':

        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        expanded = expanded =(trialdataslice['Previous Choice']=='R')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Choice']=='L')*trialdataslice['Previous Coh']
        for q in range(2):
            for x in range(8):

                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==outcomes[q]))[0]
                    for i in range(nIters):
                        hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                except ValueError:
                    hold2[:,:,q,x,i] = np.nan

        full = np.nanmean(hold2, -1)

    return full


def bootstrapped_projset_cond2(dFFbytrial, probevar, cond, trialdataslice, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]


    alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
    cohs = [1, 0.64, 0.32, 0.16]
    outcomes = ['Hit', 'FA']
    choices = ['R','L']
    #trialdataslice[probevar] = np.random.permutation(trialdataslice[probevar])
    if cond == 'AbsDiffBin':
        alldiffs = np.arange(5)
    elif (cond == 'DiffBin') | (cond == 'SimilarityBin'):
        alldiffs = np.arange(7)

    elif (cond == 'Previous DirCoh'):
        alldiffs = alldircoh*1
    full = np.zeros([nUnits, triallen, len(alldiffs), 8])*np.nan

    if probevar == 'DirCoh':
        hold2 = np.zeros([nUnits, triallen, len(alldiffs), 8, nIters])*np.nan

        for q in range(len(alldiffs)):

            for x in range(8):
                try:
                    IDs = np.where((trialdataslice[probevar]==alldircoh[x]) & (trialdataslice[cond]==alldiffs[q]))[0]

                    for i in range(nIters):
                        hold2[:,:,q,x,i] = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,:]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'DotsCoh':
        hold2 = np.zeros([nUnits, triallen,len(alldiffs), 8, 2, nIters])*np.nan


        for q in range(len(alldiffs)):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x]) & (trialdataslice[cond]==alldiffs[q]) & (trialdataslice.CorrectSide==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Outcomes':
        hold2 = np.zeros([nUnits, triallen,len(alldiffs), 8, 2, nIters])*np.nan

        expanded = (trialdataslice['Outcomes']=='Hit')*trialdataslice['DotsCoh'] - (trialdataslice['Outcomes']=='FA')*trialdataslice['DotsCoh']

        for q in range(len(alldiffs)):
            for x in range(8):
                for ch in range(2):
                    try:

                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==alldiffs[q]) & (trialdataslice.CorrectSide==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Choice':
        hold2 = np.zeros([nUnits, triallen, len(alldiffs), 8, nIters])*np.nan

        expanded =(trialdataslice['Choice']=='R')*trialdataslice['DotsCoh'] - (trialdataslice['Choice']=='L')*trialdataslice['DotsCoh']
        for q in range(len(alldiffs)):
            for x in range(8):
                #for ch in range(2):
                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==alldiffs[q]))[0]# & (trialdataslice.CorrectSide==choices[ch]))[0]
                    for i in range(nIters):
                        hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,i]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'Previous Coh':
        hold2 = np.zeros([nUnits, triallen,len(alldiffs), 8, 2, nIters])*np.nan


        for q in range(len(alldiffs)):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice[cond]==alldiffs[q]) & (trialdataslice['Previous Correct Side']==choices[ch]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,ch,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Outcome':
        hold2 = np.zeros([nUnits, triallen, len(alldiffs), 8, 2, nIters])*np.nan

        expanded = (trialdataslice['Previous Outcome']=='Hit')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Outcome']=='FA')*trialdataslice['Previous Coh']

        for q in range(len(alldiffs)):
            for x in range(8):
                for pdc in range(2):
                    try:
                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==alldiffs[q]) & (trialdataslice['Previous Choice']==choices[pdc]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,pdc,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,pdc,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Choice':

        hold2 = np.zeros([nUnits, triallen, len(alldiffs), 8, nIters])*np.nan

        expanded = expanded =(trialdataslice['Previous Choice']=='R')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Choice']=='L')*trialdataslice['Previous Coh']
        for q in range(len(alldiffs)):
            for x in range(8):
                #for pdc in range(8):
                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==alldiffs[q]))[0]
                    for i in range(nIters):
                            hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                except ValueError:
                    hold2[:,:,q,x,i] = np.nan

        full = np.nanmean(hold2, -1)

    return full


def bootstrapped_projset_cond3(dFFbytrial, probevar, cond, trialdataslice, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]


    alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
    cohs = [1, 0.64, 0.32, 0.16]
    outcomes = ['Hit', 'FA']
    choices = ['R','L']
    #trialdataslice[probevar] = np.random.permutation(trialdataslice[probevar])
    if cond == 'SameSide':
        choices = [True,False]

    full = np.zeros([nUnits, triallen, 2, 8])*np.nan

    if probevar == 'DirCoh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        for q in range(2):

            for x in range(8):
                try:
                    IDs = np.where((trialdataslice[probevar]==alldircoh[x]) & (trialdataslice[cond]==choices[q]))[0]

                    for i in range(nIters):
                        hold2[:,:,q,x,i] = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,:]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'DotsCoh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan


        for q in range(2):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x]) & (trialdataslice[cond]==choices[q]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Outcomes':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan

        expanded = (trialdataslice['Outcomes']=='Hit')*trialdataslice['DotsCoh'] - (trialdataslice['Outcomes']=='FA')*trialdataslice['DotsCoh']

        for q in range(2):
            for x in range(8):
                for ch in range(2):
                    try:
                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==choices[q]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                    except ValueError:
                        hold2[:,:,q,x,ch,i]  = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)


    if probevar == 'Choice':
        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        expanded =(trialdataslice['Choice']=='R')*trialdataslice['DotsCoh'] - (trialdataslice['Choice']=='L')*trialdataslice['DotsCoh']
        for q in range(2):
            for x in range(8):

                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==choices[q]))[0]# & (trialdataslice.CorrectSide==choices[ch]))[0]
                    for i in range(nIters):
                        hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)

                except ValueError:
                    hold2[:,:,q,x,i]  = np.nan

        full = np.nanmean(hold2, -1)

    if probevar == 'Previous Coh':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 2, nIters])*np.nan


        for q in range(2):
            for x in range(4):
                for ch in range(2):
                    try:
                        IDs = np.where((trialdataslice[probevar]==cohs[x])& (trialdataslice[cond]==choices[q]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,ch,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,ch,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Outcome':
        hold2 = np.zeros([nUnits, triallen, 2, 8, 8, nIters])*np.nan

        expanded = (trialdataslice['Previous Outcome']=='Hit')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Outcome']=='FA')*trialdataslice['Previous Coh']

        for q in range(2):
            for x in range(8):
                for pdc in range(8):
                    try:
                        IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==choices[q]) & (trialdataslice['Previous DirCoh']==alldircoh[pdc]))[0]
                        for i in range(nIters):
                            hold2[:,:,q,x,pdc,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                    except ValueError:
                        hold2[:,:,q,x,pdc,i] = np.nan

        full = np.nanmean(np.nanmean(hold2, -1),-1)

    if probevar == 'Previous Choice':

        hold2 = np.zeros([nUnits, triallen, 2, 8, nIters])*np.nan

        expanded = expanded =(trialdataslice['Previous Choice']=='R')*trialdataslice['Previous Coh'] - (trialdataslice['Previous Choice']=='L')*trialdataslice['Previous Coh']
        for q in range(2):
            for x in range(8):

                try:
                    IDs = np.where((expanded==alldircoh[x]) & (trialdataslice[cond]==choices[q]))[0]
                    for i in range(nIters):
                        hold2[:,:,q,x,i]  = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5, replace=True),:], 1)
                except ValueError:
                    hold2[:,:,q,x,i] = np.nan

        full = np.nanmean(hold2, -1)

    return full



def shuffedbootstrap(dFFbytrial, nIters=100):
    import numpy as np
    nUnits = np.shape(dFFbytrial)[0]
    triallen = np.shape(dFFbytrial)[-1]
    nTrials = np.shape(dFFbytrial)[1]


    full = np.zeros([nUnits, triallen, 8, nIters])

    alldircoh = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]

    for i in range(nIters):
        for ch in range(8):
            IDs = np.arange(nTrials)
            full[:,:,ch, i] = np.nanmean(dFFbytrial[:,np.random.choice(IDs, 5),:], 1)

    return np.mean(full,-1)




def getAUROC(arrayA, arrayB):
    from sklearn import metrics
    import numpy as np
    start = np.floor(np.nanmin((np.nanmin(arrayA), np.nanmin(arrayB))))
    end = np.ceil(np.nanmax((np.nanmax(arrayB), np.nanmax(arrayA))))

    triallen = np.shape(arrayA)[1]
    thresholds = np.arange(start, end, 0.01)
    aucs = []

    for t in range(triallen):



        tpr = []
        fpr = []
        for val in thresholds:
            tpr.append(np.nanmean(arrayA[:,t] > val))
            fpr.append(np.nanmean(arrayB[:,t] > val))


        aucs.append(metrics.auc(fpr, tpr))

    return aucs



rootdata = '/content/drive/My Drive/rdk-2afc/2P/'
sessions = [x for x in next(os.walk(rootdata))[1]]
sessions = np.array(sessions)[np.argsort(sessions)]

outcomes = ['Hit','FA']
for sesh in range(len(sessions)):

    session = sessions[sesh]
    print(session)
    path = rootdata + session
    fulldata = np.load(path+'/fulldata.npy',allow_pickle=True)[()]
    dFF = fulldata['dFF']
    nUnits = np.shape(dFF)[0]
    starts = fulldata['roi_start'].astype('int')
    ends = fulldata['roi_end'].astype('int')
    trialdata = fulldata['trialdata'].iloc[:-1,:]
    trialdata = trialdata[((trialdata.Outcomes=='Hit') | (trialdata.Outcomes=='FA'))]#&((trialdata['Previous Outcome']=='Hit') | (trialdata['Previous Outcome']=='FA'))]
    nUnits = np.shape(dFF)[0]

    dFF = dFF[:,:int(np.percentile(ends,99))]
    trialdata = trialdata[trialdata.StimStart<int(np.percentile(ends,90))].iloc[:-1]
    nTrials = len(trialdata)

    for n in range(nUnits):
        dFF[n,:starts[n]] = np.nan
        dFF[n,ends[n]:] = np.nan

    dFF = pd.DataFrame(dFF).dropna(axis=0).to_numpy()
    nUnits = np.shape(dFF)[0]
    normdFF = StandardScaler().fit_transform(dFF.transpose()).transpose()
    
    triallen = 60

    trialdata['First Lick'] = np.floor(np.nanmin((trialdata['firstLick_L'], trialdata['firstLick_R']), axis=0)*10+trialdata['StimStart'])
    trialdata['Attempted'] = (trialdata['Outcomes']!='Omission')

    holdprevdiff = []
    for tr in range(nTrials):
        if (trialdata['Previous DirCoh'].iloc[tr] > 0):
            term = 'R'
        else:
            term = 'L'
        if (trialdata['Previous Coh'].iloc[tr] > 0.5):
            term2 = 'Easy'
        else:
            term2 = 'Hard'
        holdprevdiff.append(term+term2)

    trialdata['PrevDiff'] = holdprevdiff
    trialdata['Difference'] = trialdata['DirCoh']-trialdata['Previous DirCoh']
    trialdata['AbsDiff'] = abs(trialdata['Difference'])

    trialdata['Previous Hit'] = (trialdata['Previous Outcome']=='Hit')*1-(trialdata['Previous Outcome']=='FA')*1
    trialdata['MultDiff'] = trialdata['DirCoh']*trialdata['Previous DirCoh']
    trialdata['MultDiffOutcome'] = trialdata['DirCoh']*trialdata['Previous DirCoh']*trialdata['Previous Hit']

    hist, bin_edges = np.histogram(trialdata['AbsDiff'], bins=[0,0.3,0.5,1.0,1.5,2.1])
    hold = np.digitize(trialdata['AbsDiff'], bin_edges, right=False)
    trialdata['AbsDiffBin'] = hold


    hist, bin_edges = np.histogram(trialdata['Difference'], [-2,-1.5,-0.75,-0.33,0.33,0.75,1.5,2.1])
    hold = np.digitize(trialdata['Difference'], bin_edges, right=False)
    trialdata['DiffBin'] = hold

    trialdata['AbsDiffBin'] = trialdata['AbsDiffBin']-np.nanmin(trialdata['AbsDiffBin'])
    trialdata['DiffBin'] = trialdata['DiffBin']-np.nanmin(trialdata['DiffBin'])

    dFFbytrial_stim = np.zeros([nUnits, (nTrials),triallen])*np.nan
    dFFbytrial_outcome = np.zeros([nUnits, (nTrials),triallen])*np.nan

    for tr in range(nTrials):
        if trialdata.Outcomes.iloc[tr]!='Omission':
            dFFbytrial_outcome[:,tr,:] = normdFF[:,int(trialdata.iloc[tr]['OutcomeTime'])-20:int(trialdata.iloc[tr]['OutcomeTime'])+40] 

        dFFbytrial_stim[:,tr,:] = normdFF[:,int(trialdata.iloc[tr]['StimStart'])-20:int(trialdata.iloc[tr]['StimStart'])+40]

    nIters = 100
    testvars = ['DirCoh', 'Choice', 'Outcomes', 'DotsCoh']#, 'Previous DirCoh', 'Previous Coh', 'Previous Choice']
    nAxes = 5

    allQs = np.zeros([nIters, nUnits, nAxes])
    alltestsetmeans = []
    allselectivities = np.zeros([nIters, nUnits, nAxes])
    trainsetlist = []
    testsetlist = []
    allflipaxis = np.zeros([nIters, nAxes])


    allmeanvar_stim_byabsdiff = np.zeros([nIters, len(testvars), nUnits, triallen, 5,2,8])
    allmeanvar_outcome_byabsdiff = np.zeros([nIters, len(testvars), nUnits, triallen, 5,2,8])

    oc=0
    for i in range(nIters):

        if i % 5 == 0:
            print(i)

        train_idx, test_idx = train_test_split(np.arange(len(trialdata)), test_size=0.5, stratify=trialdata.DirCoh, random_state=i*10)

        for k in range(len(testvars)):

            subset = np.where((trialdata.iloc[test_idx]['Previous Outcome']=='Hit'))[0]

            meanvar_stim_byabsdiff = bootstrapped_projset_cond2(dFFbytrial_stim[:,test_idx[subset],:], testvars[k], 'AbsDiffBin', trialdata.iloc[test_idx].iloc[subset])
            meanvar_outcome_byabsdiff = bootstrapped_projset_cond2(dFFbytrial_outcome[:,test_idx,:], testvars[k], 'AbsDiffBin', trialdata.iloc[test_idx].iloc[subset])

            allmeanvar_stim_byabsdiff[i,k,:,:,:,oc,:]  = meanvar_stim_byabsdiff
            allmeanvar_outcome_byabsdiff[i,k,:,:,:,oc,:]  = meanvar_outcome_byabsdiff


    np.save(path+'/allmeanvar_outcome_byabsdiff_2.npy', allmeanvar_outcome_byabsdiff)
    np.save(path+'/allmeanvar_stim_byabsdiff_2.npy', allmeanvar_stim_byabsdiff)

