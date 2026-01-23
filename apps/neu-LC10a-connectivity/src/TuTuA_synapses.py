#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Juliana Rhee
@ Filename: TuTuA_synapses.py
@ Create Time: 2026-01-23 10:53:42
@ Modified by: Juliana Rhee
@ Modified time: 2026-01-23 10:53:48
@ Description: Get the synapses of the TuTuA neurons to LC10a neurons.

'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# import neuprint stuff
import neuprint as neu
import neuprint_funcs as npf
import plotting as putil

import importlib
#%%
dataset = 'male-cns:v0.9'
c = npf.get_neuprint_client(dataset=dataset)
version = c.fetch_version()
figid = f'{dataset}_{version}'
print(figid)

#%% Plot style
plot_style = 'white'
min_fontsize = 12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%% Output dir
rootdir = '/Volumes/Juliana/connectome'
output_dir = os.path.join(rootdir, 'analyses', 'neuprint', 'TuTuA_synapses')

# Make output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Output directory: {output_dir}')

# %%
# Get all TuTuA neurons
