

class sessionGLM:
    def __init__(self, path):

        from numpy.lib.stride_tricks import sliding_window_view
        from numpy.matlib import repmat
        import pandas as pd
        from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

        self.fulldata = np.load(path + '/fulldata.npy', allow_pickle=True)[()]
        self.dFF = self.fulldata['dFF']
        self.nUnits = np.shape(self.dFF)[0]
        self.trialdata = self.fulldata['trialdata']
        self.trialdata = self.trialdata[(self.trialdata.Outcomes == 'Hit') | (self.trialdata.Outcomes == 'FA')]
        self.trialdata = self.trialdata.iloc[:-1]
        self.nTrials = len(self.trialdata)
        self.nUnits = np.shape(self.dFF)[0]
        self.triallen = 60

        starts = self.fulldata['roi_start'].astype('int')
        ends = self.fulldata['roi_end'].astype('int')
        for n in range(self.nUnits):
            self.dFF[n, :starts[n]] = np.nan
            self.dFF[n, ends[n]:] = np.nan

        normdFF = MinMaxScaler().fit_transform(self.dFF.transpose()).transpose()

        self.dFFbytrial = np.zeros([self.nUnits, self.nTrials, self.triallen]) * np.nan

        for tr in range(self.nTrials):
            self.dFFbytrial[:, tr, :] = normdFF[:, int(self.trialdata.iloc[tr]['StimStart']) - 10
                                                   :int(self.trialdata.iloc[tr]['StimStart']) + (self.triallen - 10)]

        self.fullDM, self.varnames, self.varidx, self.outcomelags, self.evalepochs = self.assembleDM()

    def assembleDM(self, mean_centered=True):

        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, MinMaxScaler

        nTimepoints = self.nTrials * self.triallen
        fullDM = np.zeros([518, nTimepoints])

        ### Motor parameters
        speedvar = RobustScaler(unit_variance=True).fit_transform(
            np.nanmean(sliding_window_view(self.fulldata['wheel'], window_shape=5), axis=1).reshape(-1, 1))[:, 0]
        Rlicks = RobustScaler(unit_variance=True).fit_transform(
            np.nanmean(sliding_window_view(self.fulldata['R_licks'], window_shape=5), axis=1).reshape(-1, 1))[:, 0]
        Llicks = RobustScaler(unit_variance=True).fit_transform(
            np.nanmean(sliding_window_view(self.fulldata['L_licks'], window_shape=5), axis=1).reshape(-1, 1))[:, 0]

        motorstim = np.zeros([3, self.nTrials, self.triallen]) * np.nan
        for tr in range(self.nTrials):
            motorstim[0, tr, :] = speedvar[int(self.trialdata.iloc[tr]['StimStart']) - 10:int(
                self.trialdata.iloc[tr]['StimStart']) + (self.triallen - 10)]
            motorstim[1, tr, :] = Rlicks[int(self.trialdata.iloc[tr]['StimStart']) - 10:int(
                self.trialdata.iloc[tr]['StimStart']) + (self.triallen - 10)]
            motorstim[2, tr, :] = Llicks[int(self.trialdata.iloc[tr]['StimStart']) - 10:int(
                self.trialdata.iloc[tr]['StimStart']) + (self.triallen - 10)]

        ### Params
        stim = ((self.trialdata.CorrectSide == 'R') * 1 + ((self.trialdata.CorrectSide == 'L') * -1)).to_numpy()
        coherence = self.trialdata.DotsCoh.to_numpy()
        dircoh = stim * coherence

        outcomes = (((self.trialdata.Outcomes == 'Hit') * 1) + ((self.trialdata.Outcomes == 'FA') * -1)).to_numpy()
        choices = ((self.trialdata.Choice == 'R') * 1 + ((self.trialdata.Choice == 'L') * -1)).to_numpy()
        sidecoh = choices * coherence
        conf = outcomes * coherence

        # HitCoh = ((self.trialdata.Outcomes=='Hit')*1*abs(self.trialdata.DotsCoh.to_numpy())).to_numpy()
        # FACoh = ((self.trialdata.Outcomes=='FA')*1*abs(self.trialdata.DotsCoh.to_numpy())).to_numpy()

        prevchoice = ((self.trialdata['Previous Choice'] == 'R') * 1 + (
                    self.trialdata['Previous Choice'] == 'L') * -1).to_numpy()
        prevoutcome = ((self.trialdata['Previous Outcome'] == 'Hit') * 1 + (self.trialdata['Previous Outcome'] == 'FA') * -1).to_numpy()  # *self.trialdata['Previous DirCoh']).to_numpy()
        prevcoh = abs(self.trialdata['Previous Coh'].to_numpy())
        prevdir = ((self.trialdata['Previous Correct Side'] == 'R') * 1 + (
                self.trialdata['Previous Correct Side'] == 'L') * -1).to_numpy()
        prevpe = prevoutcome * prevcoh
        prevsidecoh = prevchoice * prevcoh
        prevdircoh = prevdir * prevcoh
        prevdirsalience = abs(prevcoh - prevoutcome) * prevchoice

        # evigain = abs(self.trialdata['DotsCoh'].to_numpy())-abs(self.trialdata['Previous Coh'].to_numpy())

        dm_varID = {
            'ITITime': np.arange(10),
            'StimTime': np.arange(10, 25),
            'ChoiceTime': np.arange(25, 40),
            'OutcomeTime': np.arange(40, 75),
            'Coherence': np.arange(75, 125),
            'StimDirCoh': np.arange(125, 175),
            'OutcomeCoh': np.arange(175, 225),
            'ChoiceCoh': np.arange(225, 275),
            'PrevCoh': np.arange(275, 335),
            'PrevDirCoh': np.arange(335, 395),
            'PrevOutcomeCoh': np.arange(395, 455),
            'PrevChoiceCoh': np.arange(455, 515),
            'Speed': np.array([515]),
            'LicksR': np.array([516]),
            'LicksL': np.array([517]),
        }

        varnames = ['ITITime', 'StimTime', 'ChoiceTime', 'OutcomeTime',
                    'Coherence', 'StimDirCoh','OutcomeCoh','ChoiceCoh',
                    'PrevCoh', 'PrevDirCoh', 'PrevOutcomeCoh', 'PrevChoiceCoh',
                    'Speed', 'LicksR', 'LicksL']

        evalepochs = {'ITITime': ['ITI', 'Full'],
                      'StimTime': ['Stim', 'Full'],
                      'ChoiceTime': ['Choice', 'Full'],
                      'OutcomeTime': ['Outcome', 'Full'],
                      'Coherence': ['Stim', 'Choice', 'Outcome', 'Full'],
                      'StimDirCoh': ['Stim', 'Choice', 'Outcome', 'Full'],
                      'OutcomeCoh': ['Stim', 'Choice', 'Outcome', 'Full'],
                      'ChoiceCoh': ['Stim', 'Choice', 'Outcome', 'Full'],
                      'PrevCoh': ['ITI','Stim', 'Choice', 'Outcome', 'Full'],
                      'PrevDirCoh': ['ITI','Stim', 'Choice', 'Outcome', 'Full'],
                      'PrevOutcomeCoh': ['ITI','Stim', 'Choice', 'Outcome', 'Full'],
                      'PrevChoiceCoh': ['ITI', 'Stim', 'Choice', 'Outcome', 'Full'],
                      'Speed': ['Full'],
                      'LicksR': ['Full'],
                      'LicksL': ['Full']
                      }

        ### Fill design matrix

        outcomelags = []
        for tr in range(self.nTrials):

            for k in range(10):
                fullDM[k, (tr * self.triallen) + k] = 1

            for k in range(15):
                fullDM[k + 10, (tr * self.triallen) + 10 + k] = 1

            outcomelag = int(self.trialdata.iloc[tr]['OutcomeTime'] - self.trialdata.iloc[tr]['StimStart'])
            outcomelags.append(outcomelag + 10)

            for k in range(outcomelag - 15):
                fullDM[25 + k, (tr * self.triallen) + 25 + k] = 1

            for k in range(self.triallen - outcomelag - 10):
                fullDM[40 + k, (tr * self.triallen) + 10 + outcomelag + k] = 1

            for k in range(self.triallen - 10):
                # STIMULUS STUFF
                fullDM[75 + k, (tr * self.triallen) + 10 + k] = coherence[tr]
                fullDM[125 + k, (tr * self.triallen) + 10 + k] = dircoh[tr]
                fullDM[175 + k, (tr * self.triallen) + 10 + k] = conf[tr]
                fullDM[225 + k, (tr * self.triallen) + 10 + k] = sidecoh[tr]

            for k in range(self.triallen):

                fullDM[275 + k, (tr * self.triallen) + k] = prevcoh[tr]
                fullDM[335 + k, (tr * self.triallen) + k] = prevdircoh[tr]
                fullDM[395 + k, (tr * self.triallen) + k] = prevpe[tr]
                fullDM[455 + k, (tr * self.triallen) + k] = prevsidecoh[tr]

            fullDM[515, (tr * self.triallen):(tr * self.triallen) + self.triallen] = motorstim[0, tr, :]  # Speed
            fullDM[516, (tr * self.triallen):(tr * self.triallen) + self.triallen] = motorstim[1, tr, :]  # LicksR
            fullDM[517, (tr * self.triallen):(tr * self.triallen) + self.triallen] = motorstim[2, tr, :]  # LicksL

        varidx = dm_varID

        if mean_centered:
            fullDM = fullDM - np.mean(fullDM, 1).reshape(-1, 1)

        return fullDM, varnames, varidx, np.array(outcomelags), evalepochs

    def getepochs(self, trialIDs, epoch='Stim'):
        # Epochs: ITI / Stim / Choice / Outcome

        for x in range(len(trialIDs)):
            if epoch == 'Stim':
                holdtimepoints = np.arange(x * self.triallen + 10, x * self.triallen + 25)
            elif epoch == 'Choice':
                holdtimepoints = np.arange(x * self.triallen + 25, x * self.triallen + self.outcomelags[trialIDs[x]])
            elif epoch == 'Outcome':
                holdtimepoints = np.arange(x * self.triallen + self.outcomelags[trialIDs[x]], x * self.triallen +
                                           (self.triallen - self.outcomelags[trialIDs[x]]))
            elif epoch == 'ITI':
                holdtimepoints = np.arange(x * self.triallen, x * self.triallen + 10)
            elif epoch == 'Full':
                holdtimepoints = np.arange(x * self.triallen, (x + 1) * self.triallen)

            if x == 0:
                selecttimepoints = holdtimepoints
            else:
                selecttimepoints = np.hstack((selecttimepoints, holdtimepoints))

        return selecttimepoints

    def runGLMfit(self, nIters=25, nFolds=3):

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, MinMaxScaler
        from sklearn.linear_model import RidgeCV, SGDRegressor, ElasticNetCV, TweedieRegressor
        from sklearn.model_selection import cross_val_score, cross_validate
        from sklearn.model_selection import ShuffleSplit
        from sklearn.model_selection import StratifiedKFold
        import pandas as pd

        from tqdm.auto import tqdm

        nPred = np.shape(self.fullDM)[0]
        nVars = len(self.varnames)

        skf = StratifiedKFold(n_splits=nFolds, shuffle=True)

        betas = np.zeros([nIters, nFolds, self.nUnits, nPred])
        resultslist = []

        for i in tqdm(range(nIters), desc='Iteration', position=0, leave=True):

            print(i)
            fold = 0
            for train_index, test_index in skf.split(self.trialdata, self.trialdata['DirCoh'].astype('str')):
                # print("TRAIN:", train_index, "TEST:", test_index)

                X_train = np.zeros([len(train_index) * self.triallen, nPred])
                X_test = np.zeros([len(test_index) * self.triallen, nPred])

                for idx in range(len(train_index)):
                    X_train[(idx * self.triallen):(idx + 1) * self.triallen, :] = self.fullDM[:,
                                                                                  train_index[idx] * self.triallen:(
                                                                                                                               train_index[
                                                                                                                                   idx] + 1) * self.triallen].T

                for idx in range(len(test_index)):
                    X_test[(idx * self.triallen):(idx + 1) * self.triallen, :] = self.fullDM[:,
                                                                                 test_index[idx] * self.triallen:(
                                                                                                                             test_index[
                                                                                                                                 idx] + 1) * self.triallen].T

                y_train_ALL = np.reshape(self.dFFbytrial[:, train_index, :],
                                         (self.nUnits, len(train_index) * self.triallen))
                y_test_ALL = np.reshape(self.dFFbytrial[:, test_index, :],
                                        (self.nUnits, len(test_index) * self.triallen))

                # with tqdm(total=self.nUnits, position=1, leave=True, desc=str(fold)+' Neuron') as pbar:
                for n in tqdm(range(self.nUnits), position=1, leave=True):
                    y_train = y_train_ALL[n, :][~np.isnan(y_train_ALL[n, :])]
                    X_train_ = X_train[~np.isnan(y_train_ALL[n, :]), :]

                    y_test = y_test_ALL[n, :]
                    # X_test_ = X_test[~np.isnan(y_test_ALL[n,:]),:]

                    model = SGDRegressor(fit_intercept=True, penalty='elasticnet', l1_ratio=0.01, n_iter_no_change=10)
                    model.fit(X_train_, y_train)

                    betas[i, fold, n, :] = model.coef_

                    epochs = ['Stim', 'Choice', 'Outcome', 'ITI', 'Full']

                    testnans = np.arange(len(y_test_ALL[n, :]))[np.where(~np.isnan(y_test_ALL[n, :]))[0]]
                    for ep in range(len(epochs)):
                        epoch = epochs[ep]

                        temp = dict()
                        temp['Iteration'] = i
                        temp['Fold'] = fold
                        temp['Train Score'] = model.score(X_train_, y_train)
                        temp['Test Score'] = model.score(X_test[~np.isnan(y_test_ALL[n, :]), :],
                                                         y_test[~np.isnan(y_test_ALL[n, :])])
                        temp['ROI'] = n
                        temp['Epoch'] = epoch

                        for v in range(nVars):
                            zeroedvar = X_test * 1
                            zeroedvar[:, self.varidx[self.varnames[v]]] = np.mean(
                                zeroedvar[:, self.varidx[self.varnames[v]]])

                            subsample = self.getepochs(test_index, epoch=epoch)
                            if len(np.intersect1d(subsample, testnans)) > 1:
                                intact = model.score(X_test[np.intersect1d(subsample, testnans), :],
                                                     y_test[np.intersect1d(subsample, testnans)])

                                temp['FULL'] = intact
                                if np.isin(epoch, self.evalepochs[self.varnames[v]]):
                                    temp[self.varnames[v]] = intact - model.score(
                                        zeroedvar[np.intersect1d(subsample, testnans), :],
                                        y_test[np.intersect1d(subsample,
                                                              testnans)])

                        resultslist.append(temp)

                fold = fold + 1

        results = pd.DataFrame.from_records(resultslist)

        allresults = dict()
        allresults['Fit Results'] = results
        allresults['Betas'] = betas
        allresults['Varnames'] = self.varnames
        allresults['Var IDs'] = self.varidx

        return allresults


import os
import sys
import numpy as np

rootdata = '/Users/ningleow/Library/CloudStorage/GoogleDrive-ningleow@mit.edu/My Drive/rdk-2afc/2P/'
sessions = [x for x in next(os.walk(rootdata))[1]]
sessions = np.array(sessions)[np.argsort(sessions)]

sesh = 16
session = sessions[sesh]
print(session)
path = rootdata + session
glm = sessionGLM(path)
results = glm.runGLMfit(nIters=100, nFolds=3)

np.save(path + 'fitresults_sesh'+str(sesh)+'.npy', results)


