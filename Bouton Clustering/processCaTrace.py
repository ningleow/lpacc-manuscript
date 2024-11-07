def processCaTrace(trace, duration, finalHz=10):
    # Input:
        # trace: nUnits x framesRecorded (neuropil subtracted)
        # duration: of imaging, in seconds

    # This will 1) Smooth the trace
            #   2) Resample to finalHz
            #   3) get dF/F

    import numpy as np
    from scipy import interpolate
    from scipy import stats

    nUnits = np.shape(trace)[0]
    nFramesrec = np.shape(trace)[1]

    originalHz = nFramesrec/duration
    finalframes = finalHz*duration

    ratio = finalframes/nFramesrec
    xnew = np.arange(0, nFramesrec, 1/ratio)

    dFF = np.zeros([nUnits,finalframes])

    # 1) Smooth the trace
    x_vals = np.arange(0,10,0.5)
    sigma = 2
    halfgausskernel = np.exp(-(x_vals) ** 2 / (2 * sigma ** 2))


    for i in range(nUnits):
        trace_i = np.nan_to_num(np.convolve(trace[i,:], halfgausskernel))[:nFramesrec]
        interfx = interpolate.interp1d(np.arange(0,nFramesrec), trace_i, bounds_error=False)
        try:
            trace2 = np.nan_to_num(interfx(xnew))
        except:
            trace2 = np.nan_to_num(interfx(xnew[:-1]))

        valrange = np.linspace(0,np.nanmax(trace2),100)
        kernel = stats.gaussian_kde(trace2)
        kdepdf = kernel.evaluate(valrange)
        baseline = valrange[np.argsort(kdepdf)[-1]]

        dFF[i,:] = np.nan_to_num((trace2-baseline)/baseline)[:finalframes]

        if np.percentile(dFF[i,:],5) < 0:
            dFF[i,:] = dFF[i,:]-np.percentile(dFF[i,:],5)

    return dFF
