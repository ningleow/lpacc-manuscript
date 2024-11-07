def getWeightedActivity(rawF, rawNeuropil, clusters):
    nClusters = len(np.unique(clusters))

    nUnits = np.shape(rawF)[0]
    nFrames = np.shape(rawF)[1]

    weightedActivity = np.zeros([nClusters, nFrames])
    weightedNeuropil = np.zeros([nClusters, nFrames])

    meanF = np.mean(rawF, 1)

    for x in range(nClusters):

        subset = np.arange(nUnits)[clusters==x+1]
        nSubROIs = len(subset)

        weights = meanF[subset]/np.sum(meanF[subset])

        for k in range(nSubROIs):
            weightedActivity[x,:] = weightedActivity[x,:] + weights[k]*rawF[subset[k],:]
            weightedNeuropil[x,:] = weightedNeuropil[x,:] + weights[k]*rawNeuropil[subset[k],:]

    return weightedActivity, weightedNeuropil
