def getPerformance(trialdata, output='pStandard'):
    ### output = 'Count' / 'Accuracy' / 'pRight' / 'pLeft' / 'pStandard'
    ### pStandard is Dir 0 - Choose R

    import numpy as np
    import pandas as pd

    performance = pd.DataFrame()
    allCoh = [1,0.64,0.32,0.16]
    allDir = [0,180]

    if output == 'pStandard':
        # we want to determine a common contingency
        if len(trialdata[(trialdata['DotsDir']==0) & (trialdata['CorrectSide']=='R')]) > 0:
            contingency = 'A'
        else:
            contingency = 'B'

    if len(trialdata[trialdata['DotsCoh'] == 0.16]) == 0:
        allCoh = [1,0.64,0.32]

    for d in range(len(allDir)):
        for c in range(len(allCoh)):
            hold = trialdata[(trialdata['DotsDir']==allDir[d]) & (trialdata['DotsCoh']==allCoh[c])]

            perf = dict()
            if output == 'pStandard':
                if d == 0:
                    perf['Coh'] = allCoh[c]
                else:
                    perf['Coh'] = -allCoh[c]
            else:
                perf['Dir'] = allDir[d]
                perf['Coh'] = allCoh[c]

            if output == 'Count':
                perf['Hits'] = len(hold[hold.Outcomes=='Hit'])
                perf['FA']= len(hold[hold.Outcomes=='FA'])
                perf['Omissions'] = len(hold[hold.Outcomes=='Omission'])

            if output == 'Accuracy':
                perf['Hits'] = len(hold[hold.Outcomes=='Hit']) / (len(hold[hold.Outcomes=='FA'])+len(hold[hold.Outcomes=='Hit']))
                perf['FA']= len(hold[hold.Outcomes=='FA']) / (len(hold[hold.Outcomes=='FA'])+len(hold[hold.Outcomes=='Hit']))

            if output == 'pRight':
                perf['pRight'] = len(hold[hold.Choice=='R']) / (len(hold[hold.Choice=='R'])+len(hold[hold.Choice=='L']))

            if output == 'pLeft':
                perf['pLeft'] = len(hold[hold.Choice=='L']) / (len(hold[hold.Choice=='R'])+len(hold[hold.Choice=='L']))

            if output == 'pStandard':
                if contingency == 'A':
                    perf['pStandard'] = len(hold[hold.Choice=='R']) / (len(hold[hold.Choice=='R'])+len(hold[hold.Choice=='L']))
                elif contingency == 'B':
                    perf['pStandard'] = len(hold[hold.Choice=='L']) / (len(hold[hold.Choice=='R'])+len(hold[hold.Choice=='L']))

            performance = performance.append(perf, ignore_index=True)

    return performance
