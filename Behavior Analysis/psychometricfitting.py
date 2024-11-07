# Use psignifit to fit psychometric curves to behavioral data

def psychfit(trialset):

    import numpy as np
    import pandas as pd
    import psignifit as ps

    hold = pd.DataFrame(columns=[-1,-0.64,-0.32,-0.16,0.16, 0.32,0.64,1])
    hold = hold.append(trialset.groupby('DirCoh').ChooseR.sum())


    hold2 = pd.DataFrame(columns=[-1,-0.64,-0.32,-0.16,0.16, 0.32,0.64,1])
    hold2 = hold2.append(trialset.groupby('DirCoh').ChooseR.count())

    dat = np.zeros([8,3])
    dat[:,0] = [-1,-0.64,-0.32,-0.16,0.16,0.32,0.64,1]
    dat[:,1] = hold.to_numpy() #trialset.groupby('DirCoh').ChooseR.sum().to_numpy()
    dat[:,2] = hold2.to_numpy() #trialset.groupby('DirCoh').ChooseR.count().to_numpy()

    means = dat[:,1]/dat[:,2]

    options                = dict()   # initialize as an empty dict
    options['sigmoidName'] = 'norm'   # choose a cumulative Gauss as the sigmoid
    options['expType']     = 'YesNo'   # choose 2-AFC as the paradigm of the experiment

    results = ps.psignifit(dat, options)

    fit = results['Fit']
    data = results['data']
    options = results['options']

    minStim = min(data[:,0])
    maxStim = max(data[:,0])
    stimRange = [1.1*minStim - .1*maxStim, 1.1*maxStim - .1*minStim]

    x = np.linspace(stimRange[0], stimRange[1], 1000)
    y = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1])

    return y, means

    #####################