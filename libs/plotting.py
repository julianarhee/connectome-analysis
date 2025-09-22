#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/05/09 10:17:48
@Author  :   julianarhee 
'''

import pylab as pl
import seaborn as sns

## generic
# ----------------------------------------------------------------------
# Visualization 
# ----------------------------------------------------------------------
def label_figure(fig, fig_id, x=0.01, y=0.98):
    fig.text(x, y, fig_id, fontsize=8)


def set_sns_style(style='dark', min_fontsize=6):
    font_styles = {
                    'axes.labelsize': min_fontsize+1, # x and y labels
                    'axes.titlesize': min_fontsize+1, # axis title size
                    'figure.titlesize': min_fontsize+4,
                    'xtick.labelsize': min_fontsize, # fontsize of tick labels
                    'ytick.labelsize': min_fontsize,  
                    'legend.fontsize': min_fontsize,
                    'legend.title_fontsize': min_fontsize+1
        }
    for k, v in font_styles.items():
        pl.rcParams[k] = v

    pl.rcParams['axes.linewidth'] = 0.5

    if style=='dark':
        custom_style = {
                    'axes.labelcolor': 'white',
                    'axes.edgecolor': 'white',
                    'grid.color': 'gray',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'text.color': 'white',
                    'axes.facecolor': 'black',
                    'axes.grid': False,
                    'figure.facecolor': 'black'}
        custom_style.update(font_styles)

#        pl.rcParams['figure.facecolor'] = 'black'
#        pl.rcParams['axes.facecolor'] = 'black'
        sns.set_style("dark", rc=custom_style)
    elif style == 'white':
        custom_style = {
                    'axes.labelcolor': 'black',
                    'axes.edgecolor': 'black',
                    'grid.color': 'gray',
                    'xtick.color': 'black',
                    'ytick.color': 'black',
                    'text.color': 'black',
                    'axes.facecolor': 'white',
                    'axes.grid': False,
                    'figure.facecolor': 'white'}
        custom_style.update(font_styles)
        sns.set_style('white', rc=custom_style)

    pl.rcParams['savefig.dpi'] = 400
    pl.rcParams['figure.figsize'] = [6,4]

    pl.rcParams['svg.fonttype'] = 'none'
