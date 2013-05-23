import numpy as np
import matplotlib.pyplot as plt

def raster(data, limits=(-50,150)):
    """ Creates a raster plot from the data.  Each dash is a spike, each row
        is a trial.
        
        Arguments
        ---------
        limits : tuple : time limits in milliseconds
    """
    
    spikes = data['spikes']
    x = np.concatenate(spikes.values)*1000. # In milliseconds
    y = [ [i]*j for i, j in enumerate(spikes.map(len)) ]
    y = np.concatenate(y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, marker='|', color='k')
    ax.plot([0,0], [-1,spikes.count()], '--', color='grey')
    ax.set_xlim(*limits)
    ax.set_ylim(-1,spikes.count())
    
    return ax
    
    