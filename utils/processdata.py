import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use, get_backend
import imageio
from IPython.display import clear_output as clc
from IPython.display import display

mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"


class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out


def multiplot(yts, plot, titles = None, figsize = None, save = False, name = "multiplot"):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, list of titles, figure size, save option and save path
    """
    
    plt.figure(figsize = figsize)
    for i in range(len(yts)):
        plt.subplot(20, 20, i+1)
        plot(yts[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig(name.replace(".png", "") + ".png", transparent = True, bbox_inches='tight')


def trajectory(yt, plot, title = None, figsize = None, save = False, name = 'gif'):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, title, figure size, save option and save path
    """

    arrays = []
        
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)
        plot(yt[i])
        plt.title(title)
        plt.axis('off')
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
        

def trajectories(yts, plot, titles = None, figsize = None, save = False, name = 'gif'):
    """
    Gif of different trajectories
    Input: list of trajectories with dimensions (sequence length, data shape), plot function for a snapshot, list of titles, figure size, save option and save path
    """

    arrays = []

    for i in range(yts[0].shape[0]):

        plt.figure(figsize = figsize)
        for j in range(len(yts)):
            plt.subplot(20, 20, j+1)
            plot(yts[j][i])
            plt.title(titles[j])
            plt.axis('off')

        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
