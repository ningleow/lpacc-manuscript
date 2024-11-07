def boutonClustering(raw_F):
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.cluster.hierarchy as sch

    nUnits = np.shape(raw_F)[0]
    timepoints = np.shape(raw_F)[1]

    # downsample/bin activity in 3 bins to make correlations more robust to measurement noise
    downsample = np.zeros([nUnits,int(timepoints/3)])
    for i in range(1,int(timepoints/3)):
        downsample[:,i] = np.mean(raw_F[:,(i-1)*3:i*3],1)

    # Get pairwise correlations
    subdf = pd.DataFrame(downsample.transpose())
    X = subdf.corr().values     # correlations
    d = sch.distance.pdist(X)   # pairwise distances
    L = sch.linkage(d, method='average')

    # Find threshold for clustering based on corrrelations
    cont = True
    dist = 0.5

    while cont:
        ind = sch.fcluster(L, dist*d.max(), 'distance')

        allwithin = []
        allbetween = []
        for n in range(nUnits):
            clustid = ind[n]
            within = np.where(ind==clustid)[0]
            between = np.arange(nUnits)[np.arange(nUnits)!=n]
            within = within[within!=n]

            for w in range(len(within)):
                allwithin.append(X[n,within[w]])
                between = between[between!=within[w]]
            for b in range(len(between)):
                allbetween.append(X[n, between[b]])

        if np.min(allwithin) >= np.percentile(allbetween, 85):
            # Set a minimum criteria of having the minimum
            # correlation within clusters larger than 85 percentile correlations
            cont = False
            print('Dist = ' + str(dist))
            print('Within Min = ' + str(np.min(allwithin)))
            print('85th percentile Between = ' + str(np.percentile(allbetween, 85)))
            print('nClusters = ' + str(len(np.unique(ind))))
        else:
            dist = dist - 0.005


    clustsort = [subdf.columns.tolist()[i] for i in list((np.argsort(ind)))]
    clusters = ind
    return clusters, clustsort
