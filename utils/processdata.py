import torch
import matplotlib.pyplot as plt
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


def multiplot(yts, plot, titles = None, figsize = None):
    """
    Multi plot of different snapshots
    Input: list of snapshots and related plot function
    """
    
    plt.figure(figsize = figsize)
    for i in range(len(yts)):
        plt.subplot(20, 20, i+1)
        plot(yts[i])
        plt.title(titles[i])
        plt.axis('off')


def trajectory(yt, plot, title = None, figsize = None):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape) and related plot function for a snapshot
    """
    
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)
        plot(yt[i])
        plt.title(title)
        plt.axis('off')
        display(plt.gcf())
        plt.close()
        clc(wait=True)

def trajectories(yts, plot, titles = None, figsize = None):
    """
    Gif of different trajectories
    Input: list of trajectories with dimensions (sequence length, data shape) and plot function for a snapshot
    """
    
    for i in range(yts[0].shape[0]):

        plt.figure(figsize = figsize)
        for j in range(len(yts)):
            plt.subplot(20, 20, j+1)
            plot(yts[j][i])
            plt.title(titles[j])
            plt.axis('off')
        
        display(plt.gcf())
        plt.close()
        clc(wait=True)
