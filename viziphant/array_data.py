"""
This module contains functions to plot activity data from two-dimensional
recording arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import neo

def plot_array(
        axis, dimension, array_data, direction_data=None, electrode_mask=None, 
        color_map=None, minc=-np.pi, maxc=np.pi, 
        arrow_scale_factor=1, summary_arrow_scale_factor=None, rotation=0):
    """Plots a two-dimensional color plot of array-like data with optional 
    direction arrows (e.g., gradients) on top.

    Parameters
    ==========
    axis : int
        Handle to axis in which to plot the array.
    dimension : tuple of int of length two
        The NxM array dimensions. 
    array_data : neo.AnalogSignal
        Contains the signal array of X=NxM signals to plot on the array.
        Convention: Signal X=0 is the lower left, signal X=M-1 is the lower right,
        signal X=MxN-1 is the top right. Only a single time point of the array may be
        given, such that the dimensions of `array_data` are `(1, N*M)`. 
    direction_data : neo.AnalogSignal
        Directions to plot as arrow. The function takes complex values in the
        AnalogSignal that determined direction and magnitude of the arrows.
        Note: To plot gradients, you need to specify the reverse directions.
        If
        set to None, no arrows are plotted. Default: None
    electrode_mask : numpy.array of bool
        one-dimensional array of NxM booleans that indicate whether to plot
        data from each of the NxM electrodes. True at position i will plot the
        i-th channel in the AnalogSignal object, False will plot a black
        square. If None, all electrodes are plotted.
        Default: None
    color_map : matplotlob.pyplot.cm.get_cmap
        Colormap to for plotting the data. If None, the current color map is
        used. 
        Default: None
    arrow_scale_factor : float
        Factor by which to scale the direction arrows in each square.
        Default: 1
    summary_arrow_scale_factor : float
        Factor by which to scale the summary direction arrows in the four array
        quadrants. If None, these arrows are not plotted.
        Default: None
    rotation : float
        Rotates the array CW by `rotation` degree. Currently partly supported,
        directions are not yet rotated. Default: 0.


    Returns
    =======
    numpy.array
        Image returned by pcolormesh.
        
    Example
    =======
    # Create fake data for a 4x25 electrode array, that is the array is 4 
    # electrodes high and 25 wide. Electrode 1 is bottom left, electrode 100 
    # is top right. We set channel to a high value. Also, for simplicity, we'll
    # create only one sample of the AnalogSignal.
    >>> grid_data = np.random.random((1,100))
    >>> grid_data[0,3] = 3
    >>> grid_data_obj = neo.AnalogSignal(
    ...     grid_data*pq.mV, t_start=0*pq.s, sampling_rate=1000*pq.Hz)
    
    # Create corresponding directions in the direction 0 rad
    direction_data = np.exp(np.complex(0,1)*np.random.random((1,100))*.5)
    >>> direction_data_obj = neo.AnalogSignal(
    ...     direction_data*pq.dimensionless, 
    ...     t_start=0*pq.s, sampling_rate=1000*pq.Hz)
    >>> cmap = plt.cm.get_cmap('hsv')
    
    # Plot the data
    >>> plot_array(
    ...     plt.gca(), (4,25), grid_data_obj, direction_data_obj, 
    ...     arrow_scale_factor=.7, summary_arrow_scale_factor=.5, 
    ...     color_map=cmap)
    >>> plt.show()
    """
    if not color_map:
        color_map = plt.cm.get_cmap()
        
    # Broken electrodes
    color_map.set_bad(color='k')

    # If no electrode mask is given, select all electrodes
    if not electrode_mask:
        electrode_mask = [True] * (dimension[0]*dimension[1])
    # Arrange data on a grid
    grid_data = np.ma.array(np.reshape(
        array_data, dimension), mask=(np.equal(electrode_mask, False)))

    # Plot phases as pcolormesh
    X, Y = np.meshgrid(
        np.arange(0, dimension[1]+1), np.arange(0, dimension[0]+1))

    if rotation != 0:
        # rotate clockwise
        rotation = 2.*np.pi-rotation

        # affine transform
        X = X - (dimension[0]/2-0.5) #4.5
        Y = Y - (dimension[1]/2-0.5) #4.5
        Xr = X * np.cos(rotation) - Y * np.sin(rotation)
        Yr = X * np.sin(rotation) + Y * np.cos(rotation)
        X = Xr + (dimension[0]/2-0.5)
        Y = Yr + (dimension[0]/2-0.5)
    image = axis.pcolormesh(
        X, Y, grid_data, vmin=minc, vmax=maxc, cmap=color_map)

    # Plot arrows
    # TODO: Implement array rotation for diretions
    if direction_data is not None:
        direc_grid = np.ma.array(np.reshape(
            direction_data, dimension), mask=(electrode_mask is False))
        largedirec_grid = np.ma.array([[
            np.mean(direc_grid[
                0:int(dimension[0]/2), 0:int(dimension[1]/2)]),
            np.mean(direc_grid[
                0:int(dimension[0]/2), int(dimension[1]/2):dimension[1]])],[
            np.mean(direc_grid[
                int(dimension[0]/2):dimension[0], 0:int(dimension[1]/2)]),
            np.mean(direc_grid[
                int(dimension[0]/2):dimension[0], 
                int(dimension[1]/2):dimension[1]])]])

        # Create mesh grid for arrows
        X, Y = np.meshgrid(
            np.arange(0.5, dimension[1]+0.5), np.arange(0.5, dimension[0]+0.5))
        axis.quiver(
            X, Y, np.real(direc_grid), np.imag(direc_grid),
            color='k', units='x', 
            scale=arrow_scale_factor, scale_units='x', pivot='middle')

        # Plot summary arrows
        if summary_arrow_scale_factor:
            X, Y = np.meshgrid([
                int(dimension[1]/4), int(dimension[1]*3/4)],[
                int(dimension[0]/4), int(dimension[0]*3/4)])
            axis.quiver(
                X, Y, np.real(largedirec_grid), np.imag(largedirec_grid),
                color='w', units='x', 
                scale=summary_arrow_scale_factor, scale_units='x', 
                pivot='middle', width=0.2, headaxislength=5)

    # Adapt axis
    axis.set_aspect('equal')
    axis.tick_params(
        left='off', bottom='off', top='off', right='off', labelsize='xx-small')
    axis.set_xticks(np.arange(0.5, dimension[1]+0.5, 1))
    axis.set_xticklabels(
       [_+1 for _ in range(dimension[1])])
    axis.set_yticks(np.arange(0.5, dimension[0]+0.5, 1))
    axis.set_yticklabels(
       [_*dimension[0]+1 for _ in range(dimension[1])])

    return image
