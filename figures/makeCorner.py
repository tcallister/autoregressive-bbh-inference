import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import style
import os
style.use(os.path.dirname(os.path.realpath(__file__))+'/plotting.mplstyle')

def getBounds(data):

    """
    Helper function to obtain 90% credible bounds from a list of samples
    Invoked by plot_corner to create labels on 1D posteriors

    Parameters
    ----------
    data : list or numpy.array
        1D array of samples

    Returns
    -------
    med : float
        Median of samples
    upperError : float
        Difference between 95th and 50th percentiles of data
    lowerError : float
        Difference between 50th and 5th percentiles of data
    """

    # Transform to a numpy arry
    data = np.array(data)

    # Get median, 5% and 95% quantiles
    med = np.median(data)
    upperLim = np.sort(data)[int(0.95*data.size)]
    lowerLim = np.sort(data)[int(0.05*data.size)]
 
    # Turn quantiles into upper and lower uncertainties
    upperError = upperLim-med
    lowerError = med-lowerLim
    
    return med,upperError,lowerError
    
def plot_corner(fig,plot_data,color,hist_alpha=0.7,bins=20,labelsize=14,logscale=False,vmax=None):

    """
    Helper function to generate corner plots of posterior samples.
    The primary input, `plot_data`, should be a nested dictionary containing data to be plotted.
    Every item in the parent dictionay corresponds to a parameter column, and should possess the following keys:

    * `data`: Posterior sample values
    * `plot_bounds`: Tuple of min/max values to display on plot
    * `label`: A latex string for axis labeling

    e.g.

    plot_data = {'mu_chi':{'data':[...],'plot_bounds':(-1,1),'label':r"$\mu_\chi$"},
                'sig_chi':{'data':[...],'plot_bounds':(0,1),'label':r"$\sigma_\chi$"}}

    Parameters
    ----------
    fig : matplotlib figure object
        Figure object to populate
    plot_data : dict
        Dictionary containing data to plot; see above
    color : str
        Hexcode defining plot color
    hist_alpha : float
        Defines transparency of 1D histograms (optional; default 0.7)
    bins : int
        Defines number of 1D histogram bins and 2D hexbins to use (optional; default 20)
    labelsize : int
        Defines fontsize of axis labels (optional; default 14)
    logscale : bool
        If true, a logarithmic color scale is adopted for 2D posteriors (optional; default False)
    vmax : None or float
        User-specified maximum for 2D colorscale (optional; default None)

    Returns
    -------
    fig : matplotlib figure
        Populated figure
    """
    
    if logscale==True:
        hexscale='log'
    else:
        hexscale=None

    # Define a linear color map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color])
    
    # Loop across dimensions that we want to plot
    keys = list(plot_data)    
    ndim = len(keys)
    for i,key in enumerate(keys):
       
        # Plot the marginal 1D posterior (i.e. top of a corner plot column)
        ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i))
        
        ax.hist(plot_data[key]['data'],bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],bins),\
               rasterized=True,color=color,alpha=hist_alpha,density=True,zorder=0)
        ax.hist(plot_data[key]['data'],bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],bins),\
                histtype='step',color='black',density=True,zorder=2)
        ax.grid(True,dashes=(1,3))
        ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])
        ax.set_title(r"${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(*getBounds(plot_data[key]['data'])),fontsize=14)

        # Turn off tick labels if this isn't the first dimension
        if i!=0:
            ax.set_yticklabels([])

        # If this is the last dimension add an x-axis label
        if i==ndim-1:
            ax.set_xlabel(plot_data[key]['label'],fontsize=labelsize)
            
        # If not the last dimension, loop across other variables and fill in the rest of the column with 2D plots
        else:
            
            ax.set_xticklabels([])
            for j,k in enumerate(keys[i+1:]):
                
                # Make a 2D density plot
                ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i + (j+1)*ndim))
                
                ax.hexbin(plot_data[key]['data'],plot_data[k]['data'],cmap=cmap,mincnt=1,gridsize=bins,bins=hexscale,\
                         rasterized=True,extent=(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],plot_data[k]['plot_bounds'][0],plot_data[k]['plot_bounds'][1]),
                         linewidths=(0,),zorder=0,vmax=vmax)
                
                # Set plot bounds
                ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])
                ax.set_ylim(plot_data[k]['plot_bounds'][0],plot_data[k]['plot_bounds'][1])
                ax.grid(True,dashes=(1,3))
                
                # If still in the first column, add a y-axis label
                if i==0:
                    ax.set_ylabel(plot_data[k]['label'],fontsize=labelsize)
                else:
                    ax.set_yticklabels([])
               
                # If on the last row, add an x-axis label
                if j==ndim-i-2:
                    ax.set_xlabel(plot_data[key]['label'],fontsize=labelsize)
                else:
                    ax.set_xticklabels([])
                    
    plt.tight_layout()    
    return fig
