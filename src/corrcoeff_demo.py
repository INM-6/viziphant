# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:37:02 2016

@author: pietro
"""
# =============================================================================
# Initialization
# =============================================================================

import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

# provides neo framework and I/Os to load ind and nest data
import neo

# provides core analysis library component
import elephant

import h5py_wrapper.wrapper
inch2cm = 0.3937        # conversion from inches to centimeters
label_size = 10
text_size = 16
tick_size = 8
fig_corrcoeff=plt.figure(figsize=(25*inch2cm, 25*inch2cm), dpi=10000)
fig_corrcoeff.subplots_adjust(top=.92, left=.3, right=.90, bottom=.2, hspace=0.05, wspace=0.2)
for idx, data in enumerate(['spinnaker', 'nest']):
    filename =  '../../results/release_demo/viz_corrcoeff_' + data + '.h5'
    cc = h5py_wrapper.wrapper.load_h5(filename)
    # Plot corrcoeff
    ax_corrcoeff = plt.subplot2grid(
        (2, 2), (0, idx), rowspan=1, colspan=1, aspect=1)
    pcol_corrcoeff=ax_corrcoeff.pcolor(cc['original_measure'], vmax=0.05, cmap=plt.cm.seismic)
    ax_corrcoeff.set_title(data, size=text_size)
    ax_corrcoeff.set_xticklabels(())
    if idx == 1:
        ax_corrcoeff.set_yticklabels(())
        x, y, dx, dy = ax_corrcoeff.get_position().bounds
        cbar_ax = plt.axes((x+0.3, y+0.03, 0.02, 0.28))
        cbar = fig_corrcoeff.colorbar(pcol_corrcoeff, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.text(5, .5, 'corrcoeff', va='center', ha='left', rotation=270, size=label_size)

    # Plot plvalues
    ax_pvalues = plt.subplot2grid(
        (2, 2), (1, idx), rowspan=1, colspan=1, aspect=1)
    pcol_pvalues=ax_pvalues.pcolor(cc['pvalue'], cmap=plt.cm.jet_r)
    if idx == 1:
        ax_pvalues.set_yticklabels(())
        x, y, dx, dy = ax_pvalues.get_position().bounds
        cbar_ax = plt.axes((x+0.3, y+0.03, 0.02, 0.28))
        cbar = fig_corrcoeff.colorbar(pcol_pvalues, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.text(5, .5, 'p-values', va='center', ha='left', rotation=270, size=label_size)


