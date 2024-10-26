import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def contour_plot(ax, x, t, umatrix, cmap=cm.jet, title=None, labels = [None, None], **kwargs):
    xgrid, tgrid = np.meshgrid(x / 2 / np.pi, t)
    
    cont = ax.contourf(xgrid, tgrid, umatrix, cmap=cmap, **kwargs)
    
    if labels[0] is not None:
        ax.set_xlabel(r'Space $x/2\pi$', fontsize=labels[0])
    if labels[1] is not None:
        ax.set_ylabel(r'Time $t$', fontsize=labels[1])
    
    ax.set_xlim(x[0]/2/np.pi, x[-1]/2/np.pi)
    ax.set_ylim(t[0], t[-1])
    
    if title is not None:
        ax.set_title(title)
    plt.colorbar(cont, ax=ax)
    
def plot_FOM_vs_Recon(x, t, fom: np.ndarray, recons: dict, std_recons: dict = None,
                      cmap = sns.color_palette('icefire', as_cmap=True), cmap_res = cm.hot,
                      nlevels = 30, time_idx = [0.3, 0.6, 0.9],
                      ylabel = None, filename = None, figsize=[6,5],
                      fontsize=15, format = 'svg',
                      box = None):
    
    assert len(x) == fom.shape[0]
    assert len(t) == fom.shape[1]
    
    nrows = 3
    
    keys = list(recons.keys())
    ncols = len(keys) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    
    # FOM
    levels = np.linspace(fom.min(), fom.max(), nlevels)
    contour_plot(axs[1, 0], x, t, fom.T, title=r'FOM', levels=levels, cmap=cmap, labels=[fontsize, fontsize])
    
    # Reconstructions and Residuals
    for key_i in range(len(keys)):
        contour_plot(axs[1, key_i+1], x, t, recons[keys[key_i]].T, title=keys[key_i], levels=levels, cmap=cmap, labels=[fontsize, fontsize])
        contour_plot(axs[0, key_i+1], x, t, np.abs(fom - recons[keys[key_i]]).T, title='Residual - '+keys[key_i], levels=nlevels, cmap=cmap_res, labels=[fontsize, fontsize])
    
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].grid(False)
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['left'].set_visible(False)
    axs[0, 0].spines['bottom'].set_visible(False)
    
    ## Line Plots
    assert len(time_idx) == ncols
    
    time_to_plot = [int(time * len(t)) for time in time_idx]
    
    colors = cmap(np.linspace(0.25,0.75,len(keys)+1))
    
    for kk in range(len(time_to_plot)):
        
        # FOM
        axs[2, kk].plot(x/2/np.pi, fom[:, time_to_plot[kk]], color=colors[0], label='FOM')
        
        # Reconstruction
        for key_i in range(len(keys)):
            axs[2, kk].plot(x/2/np.pi, recons[keys[key_i]][:, time_to_plot[kk]], '--', color = colors[key_i+1], label=keys[key_i])

        # Standard deviation
        if std_recons is not None:
            for key_i in range(len(keys)):
                if std_recons[keys[key_i]] is not None:
                    axs[2, kk].fill_between(x/2/np.pi, recons[keys[key_i]][:, time_to_plot[kk]] - 1.96 * std_recons[keys[key_i]][:, time_to_plot[kk]],
                                            recons[keys[key_i]][:, time_to_plot[kk]] +  1.96 * std_recons[keys[key_i]][:, time_to_plot[kk]],
                                            color=colors[key_i+1], alpha=0.3)

        axs[2, kk].set_title(r'Time $t={:.2f}$ s'.format(t[time_to_plot[kk]]), fontsize=fontsize)
        axs[2, kk].grid()
        axs[2, kk].legend(framealpha=1, fontsize=fontsize)
        axs[2, kk].set_xlabel(r'Space $x/2\pi$', fontsize=fontsize)
        axs[2, kk].set_xlim(0, max(x/2/np.pi))
        
    if ylabel is not None:
        axs[2,0].set_ylabel(ylabel, fontsize=fontsize)

    if box is not None:
        axs[0, 0].annotate(box, xy=(0.35, 0.3), xycoords='axes fraction', fontsize=fontsize,
                   bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'),
                   ha='center', va='center')


    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename+'.'+format, format=format, dpi=250, bbox_inches='tight')
    else:
        plt.show()
        
        
def plot_shred_error_bars(relative_test_errors: list[float]):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25,5))

    n_configurations = len(relative_test_errors)

    # Bars
    axs[0].bar(np.arange(1, n_configurations+1, 1), relative_test_errors, 
            color = cm.RdYlBu(np.linspace(0,1,len(relative_test_errors))),
            edgecolor='k')

    axs[0].set_xticks(np.arange(1, n_configurations+1, 1))
    axs[0].set_xlabel(r'Configurations', fontsize=20)
    axs[0].set_ylabel(r'Relative Test Error $\varepsilon_{2}$', fontsize=20)

    # Histogram
    axs[1].hist(relative_test_errors, edgecolor='k', density=True)
    axs[1].set_xlabel(r'Relative Test Error $\varepsilon_{2}$', fontsize=20)

    [ax.tick_params(axis='both', labelsize=15) for ax in axs]

    plt.tight_layout()
    
    plt.show()