import numpy as np
import matplotlib.pyplot as plt



def running_mean(x, N):

    mask=np.ones((1,N))/N
    mask=mask[0,:]
    result = np.convolve(x,mask,'same')

    return result
