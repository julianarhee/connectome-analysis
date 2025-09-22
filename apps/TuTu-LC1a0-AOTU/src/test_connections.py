#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : test_connections.py
Created        : 2025/09/21 15:53:52
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/TuTu-LC1a0-AOTU/src
Author         : jyr
Last Modified  : 
'''
#%%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard
import seaborn as sns

import neuprint_funcs as npf
import plotting as putil
import utils as util

import neuprint as neu
from neuprint import NeuronCriteria as NC

from matplotlib.lines import Line2D


#%%

def set_bokeh_line_colors(p, c='k'):
    # Customize tick colors
    p.xaxis.major_tick_line_color = c  # X-axis major ticks
    p.yaxis.major_tick_line_color = c # Y-axis major ticks
    p.xaxis.minor_tick_line_color = c  # X-axis minor ticks
    p.yaxis.minor_tick_line_color = c  # Y-axis minor ticks
    
    # Customize tick labels
    p.xaxis.major_label_text_color = c  # X-axis labels
    p.yaxis.major_label_text_color = c  # Y-axis labels
    
    # Customize the spines (borders)
    p.outline_line_color = c # Outline color
    p.xaxis.axis_line_color = c  # X-axis spine
    p.yaxis.axis_line_color = c  # Y-axis spine

    p.xaxis.axis_label_text_color = c
    p.yaxis.axis_label_text_color = c
    

#%%
# Get LC10a identities 
lc10_ids_tom = util.get_LC10_ids_Sten2021()
lc10a_ids_tom = lc10_ids_tom['LC10a'].dropna().astype(int)

print("Number of LC10a cells: ", len(lc10a_ids_tom))

from neuprint import Client
dataset = 'hemibrain:v1.2.1'
dataset_ol = 'optic-lobe:v1.1'
ol_client = Client('neuprint.janelia.org',dataset=dataset_ol, token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imp1bGlhbmEucmhlZUBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0ppOE9KZjU2c1lkNWQ0Y2NtTGhSeGNHcDhmREp6RXl0N2VKZ2x5X1FpVDIwNGFnZz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTkzODQ4NjA2NH0.RNsqAZ7V_4-M9iuJTSr_Hr7KECl4dbFnDENFZZAZIS4')
ol_client.fetch_version()

lc10a_df, lc10a_df = neu.fetch_neurons(NC(type='LC10a', client=ol_client))
lc10a_ids = lc10a_df['bodyId'].unique()
print("Number of LC10a cells: ", len(lc10a_ids))

hemi_client = Client('neuprint.janelia.org',dataset=dataset, token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imp1bGlhbmEucmhlZUBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0ppOE9KZjU2c1lkNWQ0Y2NtTGhSeGNHcDhmREp6RXl0N2VKZ2x5X1FpVDIwNGFnZz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTkzODQ4NjA2NH0.RNsqAZ7V_4-M9iuJTSr_Hr7KECl4dbFnDENFZZAZIS4')
hemi_client.fetch_version()



#%% 
# Get LC10a locations
try:
    lc10a_locs = npf.get_axo_den_locs_for_cell_ids(lc10a_ids, client=ol_client)
    print(f"✓ Retrieved locations for {lc10a_locs['cell'].nunique()} LC10a cells")
except Exception as e:
    print(f"✗ Failed to get LC10a locations: {e}")
    lc10a_locs = None

#%%
# Get all LC10a synapses
# bodyID_pre is the input cell's ID, and PRE refers to that cell's neuroT-releasing site (at its axons)
# while POST refers to the downstream target, neuroT-receiving side (bodyID_post's input sites)
confidence_thr = 0.5
try:
    syndf = neu.fetch_synapse_connections(lc10a_ids, client=hemi_client) #list(lc10a_locs['cell'].unique()))
    print(f"✓ Retrieved synapse data: {len(syndf)} connections")
    
    # Filter by confidence
    lc10a_syn = syndf[syndf['confidence_post'] > confidence_thr].copy()
    print(f"✓ Filtered to {len(lc10a_syn)} high-confidence connections")
    
except Exception as e:
    print(f"✗ Failed to fetch synapse connections: {e}")
    lc10a_syn = None


# %%

tutu_df, tutu_roi_df = neu.fetch_neurons(NC(type="TuTuA", client=hemi_client), client=hemi_client)
#%%


# %%

# Find if any lf LC10a_syn bodyId_post are in TuTuA
tutuA_1 = 676836779 # TuTuA_L 
tutaA_2 = 5813013691 # TuTuA_L
lc10a_syn_tutu = lc10a_syn[lc10a_syn['bodyId_post'].isin(tutu_df['bodyId'])]

# %%
# Connections from TuTuA to LC10a

tutu_ids = tutu_df['bodyId'].unique()
tutu_lc10a_neurons, tutu_lc10a_conn = neu.fetch_adjacencies(tutu_ids, lc10a_ids)

# Aggregate per-ROI connection weights to total connection weights
tutu_lc10a_agg = tutu_lc10a_conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

#%%
# Now, from the LC10a, find connections to AOTU19 and AOTU25
aotu25 = 892294329
aotu19 = 799868224

aotu_ids = [aotu25, aotu19]
lc10a_aotu_neurons, lc10a_aotu_conn = neu.fetch_adjacencies(lc10a_ids, aotu_ids)
lc10a_aotu_agg = lc10a_aotu_conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

#%%
#  Also get LC10a synapses to AOTU19 and AOTU25
lc10a_aotu_syn = lc10a_syn[lc10a_syn['bodyId_post'].isin(aotu_ids)]

# Find LC10a bodyId_pre that go to both aotu19 and aotu25
#lc10_aotu19 = lc10a_aotu_syn[lc10a_aotu_syn['bodyId_post'] == aotu19]['bodyId_pre'].unique()
#lc10_aotu25 = lc10a_aotu_syn[lc10a_aotu_syn['bodyId_post'] == aotu25]['bodyId_pre'].unique()
#lc10_both_aotu = set(lc10_aotu19) & set(lc10_aotu25)
#print(f"Number of LC10a neurons that go to both AOTU19 and AOTU25: {len(lc10_both_aotu)}")

#%%
# Find where POST tutu_lc10a_agg that are the PRE in lc10a_aotu_agg
tutu_lc10a_aotu = tutu_lc10a_agg[tutu_lc10a_agg['bodyId_post'].isin(lc10a_aotu_agg['bodyId_pre'])]

# %%
# Get synapses of tutu_lc10a_aotu PRE body IDs (TuTuA neurons projecting to LC10a)
tutu_lc10a_aotu_syn = neu.fetch_synapse_connections(tutu_lc10a_aotu['bodyId_pre'].unique())
tutu_lc10a_aotu_syn = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['confidence_post'] > confidence_thr]


# %%
# Get LC10a IDs that go to AOTU19 (or AOTU25)
lc10a_to_aotu19_id = lc10a_aotu_agg[lc10a_aotu_agg['bodyId_post'] == aotu19]['bodyId_pre'].unique()
lc10a_to_aotu25_id = lc10a_aotu_agg[lc10a_aotu_agg['bodyId_post'] == aotu25]['bodyId_pre'].unique()

lc10a_locs.loc[lc10a_locs['cell'].isin(lc10a_to_aotu19_id), 'aotu_type'] = 'AOTU19'
lc10a_locs.loc[lc10a_locs['cell'].isin(lc10a_to_aotu25_id), 'aotu_type'] = 'AOTU25'

# Divide TuTu into LC10a -> AOTU19 and LC10a -> AOTU25
#tutu_lc10a_aotu19_syn = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu19_id)]
#tutu_lc10a_aotu25_syn = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu25_id)]

#%%
# Count synapsesa
tutu_lc10a_aotu_syn['count_pre'] = tutu_lc10a_aotu_syn.groupby('bodyId_pre')['x_post'].transform('count')
tutu_lc10a_aotu_syn['count_post'] = tutu_lc10a_aotu_syn.groupby('bodyId_post')['x_post'].transform('count')

lc10a_aotu_syn['count_pre'] = lc10a_aotu_syn.groupby('bodyId_pre')['x_post'].transform('count')
lc10a_aotu_syn['count_post'] = lc10a_aotu_syn.groupby('bodyId_post')['x_post'].transform('count')

# Add aotu_type
lc10a_aotu_syn.loc[lc10a_aotu_syn['bodyId_post'] == aotu25, 'aotu_type'] = 'AOTU25'
lc10a_aotu_syn.loc[lc10a_aotu_syn['bodyId_post'] == aotu19, 'aotu_type'] = 'AOTU19'
# Mark common
common_lc10a_neurons = list(set(lc10a_to_aotu19_id) & set(lc10a_to_aotu25_id))
lc10a_aotu_syn.loc[lc10a_aotu_syn['bodyId_pre'].isin(common_lc10a_neurons), 'aotu_type'] = 'common'

tutu_lc10a_aotu_syn.loc[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu25_id), 'aotu_type'] = 'AOTU25'
tutu_lc10a_aotu_syn.loc[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu19_id), 'aotu_type'] = 'AOTU19'
# Mark commona
tutu_to_aotu19_ids = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu19_id)]['bodyId_post'].unique()
tutu_to_aotu25_ids = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu25_id)]['bodyId_post'].unique()
common_tutu_neurons = list(set(tutu_to_aotu19_ids) & set(tutu_to_aotu25_ids))
tutu_lc10a_aotu_syn.loc[tutu_lc10a_aotu_syn['bodyId_post'].isin(common_tutu_neurons), 'aotu_type'] = 'common'

# For each PRE cell, count the number of sites (all synapses for 1 bodyId_pre, across all POST sites)
#tutu_lc10a_aotu19_syn['count_pre'] = tutu_lc10a_aotu19_syn.groupby('bodyId_pre')['x_post'].transform('count')

# For each POST cell, count IT'S total number of synapses (across multiple LCs)
#tutu_lc10a_aotu19_syn['count_post'] = tutu_lc10a_aotu19_syn.groupby('bodyId_post')['x_post'].transform('count')

# Do same for AOTU25
#tutu_lc10a_aotu25_syn['count_pre'] = tutu_lc10a_aotu25_syn.groupby('bodyId_pre')['x_post'].transform('count')
#tutu_lc10a_aotu25_syn['count_post'] = tutu_lc10a_aotu25_syn.groupby('bodyId_post')['x_post'].transform('count')


# %%
# Plot LC10a neurons that go to BOTH aotu19 and aotu25

vidiris_two_colors = sns.color_palette('viridis', n_colors=3).as_hex()[:]
aotu_colors = {'AOTU19': vidiris_two_colors[-1], 
               'AOTU25': vidiris_two_colors[0],
               'common': 'lightgrey'}

# Check how many common LC10a neurons between TuTuA's that go to AOTU19 and AOTU25
common_lc10a_neurons = list(set(lc10a_to_aotu19_id) & set(lc10a_to_aotu25_id))
print(f"Number of common LC10a neurons between AOTU19 and AOTU25: {len(common_lc10a_neurons)}")

# Check how many of TuTu's that go to AOTU19 and AOTU25 are shared
#tutu_to_aotu19_ids = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu19_id)]['bodyId_post'].unique()
#tutu_to_aotu25_ids = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu25_id)]['bodyId_post'].unique()
common_tutu_neurons = list(set(tutu_to_aotu19_ids) & set(tutu_to_aotu25_ids))
print(f"Number of common TuTu-> LC10a between AOTU19 and AOTU25: {len(common_tutu_neurons)}")


plot_pop = 'LC10a'

if plot_pop == 'LC10a':
    plotd = lc10a_aotu_syn.copy()
    common_neurons = common_lc10a_neurons
    plot_var = 'bodyId_pre'
elif plot_pop == 'TuTu':
    plotd = tutu_lc10a_aotu_syn.copy()
    common_neurons = common_tutu_neurons
    plot_var = 'bodyId_post'

# Check LC10a that are common
fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

sns.scatterplot(data=plotd[~plotd['bodyId_post'].isin(common_neurons)], 
                ax=ax,
                x='y_pre', y='z_pre', 
                hue='aotu_type', marker='x', label='unique',
                palette=aotu_colors, alpha=0.75, legend=0)
ax.set_aspect('equal')
#ax.invert_yaxis()
#%
sns.scatterplot(data=plotd[plotd['bodyId_post'].isin(common_neurons)], 
                ax=ax,
                x='y_pre', y='z_pre', 
                hue='aotu_type',  marker = 'o',  
                palette=aotu_colors, alpha=0.3, legend=1)
ax.set_aspect('equal')
ax.invert_yaxis()

# Make custom legend showing aotu19 and aotu25 colors for x and o

marker_elements = [Line2D([0], [0], marker='x', color=aotu_colors['AOTU19'], lw=0, markersize=8, label='AOTU19 unique'),
                   Line2D([0], [0], marker='x', color=aotu_colors['AOTU25'], lw=0, markersize=8, label='AOTU25 unique'),
                   Line2D([0], [0], marker='o', color=aotu_colors['AOTU25'], lw=0, markersize=8, label='AOTU25 common'),
                   Line2D([0], [0], marker='o', color=aotu_colors['AOTU19'], lw=0, markersize=8, label='AOTU19 common')]
ax.legend(handles=marker_elements, loc='upper left', bbox_to_anchor=(1,1), frameon=False)

ax.set_title('{} post-partners in AOTU19 and AOTU25'.format(plot_pop), loc='left')

#%%
# Plot all the common LC10a neurons
n_plots = len(common_lc10a_neurons)
#n_plots = 20
nr = int(np.ceil(n_plots/6))
nc = int(np.ceil(n_plots/nr))
fig, axn = plt.subplots(nr, nc, figsize=(5, 6), sharex=True, sharey=True)
for ai, lc10a_id in enumerate(common_lc10a_neurons[0:n_plots]):
    ax = axn.flat[ai] #[ai//4, ai%4]
    plotd = lc10a_aotu_syn[lc10a_aotu_syn['bodyId_pre']==lc10a_id]
    sns.scatterplot(data=plotd,
                ax=ax,
                x='y_pre', y='z_pre', 
                hue='aotu_type', 
                size='count_post',
                palette=aotu_colors, legend=0)
    ax.set_aspect('equal')
    #ax.set_title(lc10a_id, fontsize=6)
    # Include text annotation of N synapses of each type
    n_19 = len(plotd[plotd['aotu_type'] == 'AOTU19'])
    n_25 = len(plotd[plotd['aotu_type'] == 'AOTU25'])
    ax.set_title(f'{lc10a_id}\nAOTU19: {n_19}\nAOTU25: {n_25}', transform=ax.transAxes, fontsize=6, ha='left', loc='left')

ax.invert_yaxis()

plt.subplots_adjust(hspace=0.8, wspace=0.8)

fig.suptitle('LC10a projections to both AOTU19 and AOTU25')


#%%

# COUNT SYNPSES ACROSS LC10a, and AOTU types
aotu_types = {'AOTU19': aotu19, 'AOTU25': aotu25} #'common': common_lc10a_neurons}

# Count syanpses across each TuTu->LC10a->AOTU pathway 
lc10a_aotu_syn['count_post_aggr'] = 0

tutu_list = []
for aotu_type in ['AOTU19', 'AOTU25']: #, 'common']:
    #tutu_list = []
    # LC10a that go to current aotu ID
    aotu_id = aotu_types[aotu_type]
    lc10a_to_aotu_curr = lc10a_aotu_syn[lc10a_aotu_syn['bodyId_post'] == aotu_id].copy() #['bodyId_pre'].unique()

    # For each bodyId_post in tutu_lc10a_aotu_syn_curr, add the count_post from lc10_aotu_syn
    for curr_lc10a_id, tutu_ in lc10a_to_aotu_curr.groupby('bodyId_pre'):
        lc10a_aotu_counts = tutu_['count_post'].unique()[0]
        # Add the count_post from lc10a_aotu_syn
        print(curr_lc10a_id, lc10a_aotu_counts) 

        #tutu_['count_post_lc10a'] = lc10a_aotu_counts
        tutu_['aotu_type'] = aotu_type
        tutu_['count_post_aggr'] = lc10a_aotu_counts
        tutu_list.append(tutu_)

lc10a_aotu_syn_aggr = pd.concat(tutu_list, ignore_index=True)

#%x
lc10a_aotu_syn_aggr[lc10a_aotu_syn_aggr['bodyId_post'].isin(common_lc10a_neurons)]


#%%# Normalize size of LC10a synapses to AOTU19 and AOTU25
size_min = lc10a_aotu_syn_aggr['count_post_aggr'].min()
size_max = lc10a_aotu_syn_aggr['count_post_aggr'].max()
print(f"count_post range: {size_min} to {size_max}")
print(f"Unique count_post values: {sorted(lc10a_aotu_syn_aggr['count_post_aggr'].unique())}")

# Create a function to normalize sizes consistently across all plots
def normalize_size(value, min_val, max_val, size_range=(20, 300)):
    """Normalize a value to size range based on global min/max"""
    if max_val == min_val:
        return size_range[0]
    normalized = (value - min_val) / (max_val - min_val)
    return size_range[0] + normalized * (size_range[1] - size_range[0])

# Add normalized size columns to main dataframe
lc10a_aotu_syn_aggr['norm_size'] = lc10a_aotu_syn_aggr['count_post_aggr'].apply(
    lambda x: normalize_size(x, size_min, size_max)
)

# Check the normalized size range
print(f"norm_size range: {lc10a_aotu_syn_aggr['norm_size'].min()} to {lc10a_aotu_syn_aggr['norm_size'].max()}")
print(f"Unique norm_size values: {sorted(lc10a_aotu_syn_aggr['norm_size'].unique())}")

fig, axn = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# Plot AOTU types separately with shared size mapping (no legends)
ax=axn[0]
aotu19_data = lc10a_aotu_syn_aggr[lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']
sns.scatterplot(data=aotu19_data,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU19'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=False)  # No legend for individual plots
ax.set_title('LC10a -> AOTU19\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu19, n_tutu_neurons, n_lc10a_aotu19, n_lc10a_neurons))

ax=axn[1]
aotu25_data = lc10a_aotu_syn_aggr[lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']
sns.scatterplot(data=aotu25_data,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU25'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=False)  # No legend for individual plots
ax.set_title('LC10a -> AOTU25\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu25, n_tutu_neurons, n_lc10a_aotu25, n_lc10a_neurons))
for ax in axn:
    ax.set_aspect('equal')

ax.invert_yaxis()

#%%
# Make a venn diagram of the common and unique LC10a neurons
# Label the sets and use the aotu colors
from matplotlib_venn import venn2

# Create Venn diagram
venn_diagram = venn2([set(lc10a_to_aotu19_id), set(lc10a_to_aotu25_id)])

# Apply AOTU colors to the Venn diagram
if venn_diagram.get_patch_by_id('10'):
    venn_diagram.get_patch_by_id('10').set_color(aotu_colors['AOTU19'])
    venn_diagram.get_patch_by_id('10').set_alpha(0.7)
if venn_diagram.get_patch_by_id('01'):
    venn_diagram.get_patch_by_id('01').set_color(aotu_colors['AOTU25'])
    venn_diagram.get_patch_by_id('01').set_alpha(0.7)
if venn_diagram.get_patch_by_id('11'):
    venn_diagram.get_patch_by_id('11').set_color('#800080')  # Purple for overlap
    venn_diagram.get_patch_by_id('11').set_alpha(0.7)

# Add custom labels
plt.title('LC10a Connections to AOTU19 and AOTU25')
if venn_diagram.get_label_by_id('10'):
    venn_diagram.get_label_by_id('10').set_text('AOTU19\nonly')
if venn_diagram.get_label_by_id('01'):
    venn_diagram.get_label_by_id('01').set_text('AOTU25\nonly')
if venn_diagram.get_label_by_id('11'):
    venn_diagram.get_label_by_id('11').set_text('Both\nAOTU19\n& AOTU25')
#venn2([set(lc10a_to_aotu19_id), set(lc10a_to_aotu25_id)])   
 
#%%

tutu_types = np.array([ 708290604, 5813013691,  925008763,  676836779])
n_tutu_neurons = len(tutu_types)

#tutu_types = tutu_lc10a_aotu_syn['bodyId_pre'].unique()
tutu_colors = sns.color_palette('colorblind', n_colors=len(tutu_types)).as_hex()[:]
tutu_cdict = {k: v for k, v in zip(tutu_types, tutu_colors)} 

fig, axn = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)

for i, (tutu_id, tutu_data) in enumerate(tutu_lc10a_aotu_syn.groupby('bodyId_pre')):
    ax=axn[i]
    sns.scatterplot(data=tutu_data,
                ax=ax,
                x='y_post', y='z_post', 
                hue = 'bodyId_pre',
                palette=tutu_cdict,
                #color=aotu_colors['AOTU25'], 
                #size='norm_size', 
                #sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=i==n_tutu_neurons-1)  # No legend for individual plots

    ax.set_title(f'{tutu_id}')
    ax.set_aspect('equal')

ax.invert_yaxis()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), frameon=False)



#%%
# Color LC10a BODY IDs by their dendritic location
# This is the retinotopy from Lobula

sortvar = 'y'
# get cell ids 
sorted_by_LC10a_position_ids = lc10a_locs[lc10a_locs['type']=='dendrite'].sort_values(by=sortvar)['cell'] #[0::2]
# make color map from sorted location
clist_lc10a_position = sns.color_palette('viridis', n_colors=len(sorted_by_LC10a_position_ids)).as_hex()[:]
cdict_lc10a_position = dict((k, v) for k, v in zip(sorted_by_LC10a_position_ids, clist_lc10a_position))


# %%
#aotu_colors = {'AOTU19': 'red', 'AOTU25': 'blue'}
fig, axn = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax=axn[0]
ax.set_title("TuTuA -> LC10a -> AOTU19")
sns.scatterplot(data=tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(tutu_to_aotu19_ids)], ax=ax,
                x='y_post', y='z_post', 
                hue='bodyId_post', #'count_post', 
                edgecolor='none',
                palette=cdict_lc10a_position, legend=0)
# Also plot common TuTu neurons
sns.scatterplot(data=tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(common_tutu_neurons)], ax=ax,
                x='y_post', y='z_post', 
                color='none', edgecolor='black',  alpha=1, lw=0.1, legend=0)
ax=axn[1]
ax.set_title("TuTuA -> LC10a -> AOTU25")
sns.scatterplot(data=tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(tutu_to_aotu25_ids)], ax=ax,
                x='y_post', y='z_post', 
                hue='bodyId_post', #'count_post', 
                edgecolor='none',
                palette=cdict_lc10a_position, legend=0)
# Also plot common TuTu neurons
sns.scatterplot(data=tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(common_tutu_neurons)], ax=ax,
                x='y_post', y='z_post', 
                color='none', edgecolor='black',  alpha=1, lw=0.1, legend=0)

ax=axn[2]
ax.set_title("LC10a projections to AOTU19/25")
sns.scatterplot(data=lc10a_aotu_syn, ax=ax,
                x='y_pre', y='z_pre', 
                hue='aotu_type', 
                palette=aotu_colors)
# Plot common LC10a neurons between AOTU19 and AOTU25
sns.scatterplot(data=lc10a_aotu_syn[lc10a_aotu_syn['bodyId_pre'].isin(common_lc10a_neurons)], ax=ax,
                x='y_pre', y='z_pre', 
                color='black', alpha=0.05, legend=0)
# Custom legend
marker_elements = [Line2D([0], [0], marker='o', color=aotu_colors['AOTU19'], lw=0, markersize=8, label='AOTU19'),
                   Line2D([0], [0], marker='o', color=aotu_colors['AOTU25'], lw=0, markersize=8, label='AOTU25'),
                   Line2D([0], [0], marker='o', color='black', lw=0, markersize=8, label='both')]
ax.legend(handles=marker_elements, loc='upper left', bbox_to_anchor=(1,1), frameon=False)

for ax in axn:
    ax.set_aspect('equal')
    ax.invert_yaxis()

#%%


# COUNT SYNPSES ACROSS TuTu, LC10a, and AOTU types
aotu_types = {'AOTU19': aotu19, 'AOTU25': aotu25} #'common': common_lc10a_neurons}

# Count syanpses across each TuTu->LC10a->AOTU pathway 
# (some LC10s will be counted twice since they are common)

# Add aotu_type
#lc10a_aotu_syn.loc[lc10a_aotu_syn['bodyId_post'] == aotu25, 'aotu_type'] = 'AOTU25'
#lc10a_aotu_syn.loc[lc10a_aotu_syn['bodyId_post'] == aotu19, 'aotu_type'] = 'AOTU19'

#tutu_lc10a_aotu_syn.loc[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu25_id), 'aotu_type'] = 'AOTU25'
#tutu_lc10a_aotu_syn.loc[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu19_id), 'aotu_type'] = 'AOTU19'

tutu_lc10a_aotu_syn['count_aggr'] = 0

tutu_list = []
for aotu_type in ['AOTU19', 'AOTU25']: #, 'common']:
    #tutu_list = []
    # LC10a that go to current aotu ID
    aotu_id = aotu_types[aotu_type]

    lc10a_to_aotu_curr = lc10a_aotu_syn[lc10a_aotu_syn['bodyId_post'] == aotu_id]['bodyId_pre'].unique()

    # Tutu synapses that go to this AOTU-specific LC10a subset
    tutu_lc10a_aotu_syn_curr = tutu_lc10a_aotu_syn[tutu_lc10a_aotu_syn['bodyId_post'].isin(lc10a_to_aotu_curr)]

    # For each bodyId_post in tutu_lc10a_aotu_syn_curr, add the count_post from lc10_aotu_syn
    for curr_lc10a_id, tutu_ in tutu_lc10a_aotu_syn_curr.groupby('bodyId_post'):
        # Add the count_post from lc10a_aotu_syn
        lc10a_aotu_counts = lc10a_aotu_syn[(lc10a_aotu_syn['bodyId_pre'] == curr_lc10a_id)
                                            & (lc10a_aotu_syn['bodyId_post'] == aotu_id)]['count_post'].unique()[0] #.sum()
        print(curr_lc10a_id, lc10a_aotu_counts) 

        tutu_['count_post_lc10a'] = lc10a_aotu_counts
        tutu_['aotu_type'] = aotu_type
        tutu_['count_post_aggr'] = tutu_['count_post_lc10a'] + tutu_['count_post']
        tutu_list.append(tutu_)

    #if aotu_type == 'AOTU19':
    #    tutu_aotu19_aggr = pd.concat(tutu_list, ignore_index=True)
    #else:
    #    tutu_aotu25_aggr = pd.concat(tutu_list, ignore_index=True)
tutu_lc10a_aotu_syn_aggr = pd.concat(tutu_list, ignore_index=True)

#%x
tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['bodyId_post'].isin(common_lc10a_neurons)]

#%%
# Plot TuTu synapses, with LC10a weights added

# Find TuTu neurons common to both AOTU types
aotu19_tutus = set(tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']['bodyId_pre'])
aotu25_tutus = set(tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']['bodyId_pre'])
common_tutus = aotu19_tutus.intersection(aotu25_tutus)
n_tutu_neurons = len(tutu_lc10a_aotu_syn_aggr['bodyId_pre'].unique())

n_tutu_aotu19 = len(tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']['bodyId_pre'].unique())
n_tutu_aotu25 = len(tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']['bodyId_pre'].unique())

n_lc10a_aotu19 = len(lc10a_aotu_syn[lc10a_aotu_syn['aotu_type'] == 'AOTU19']['bodyId_pre'].unique())
n_lc10a_aotu25 = len(lc10a_aotu_syn[lc10a_aotu_syn['aotu_type'] == 'AOTU25']['bodyId_pre'].unique())
n_lc10a_neurons = len(lc10a_aotu_syn['bodyId_pre'].unique())    

print(f"TuTu neurons in AOTU19: {len(aotu19_tutus)} of {n_tutu_neurons}")
print(f"TuTu neurons in AOTU25: {len(aotu25_tutus)} of {n_tutu_neurons}")
print(f"Common TuTu neurons: {len(common_tutus)}")

# Check the range of count_post_aggr values
size_min = tutu_lc10a_aotu_syn_aggr['count_post_aggr'].min()
size_max = tutu_lc10a_aotu_syn_aggr['count_post_aggr'].max()
print(f"count_post_aggr range: {size_min} to {size_max}")

# Create a function to normalize sizes consistently across all plots
def normalize_size(value, min_val, max_val, size_range=(20, 300)):
    """Normalize a value to size range based on global min/max"""
    if max_val == min_val:
        return size_range[0]
    normalized = (value - min_val) / (max_val - min_val)
    return size_range[0] + normalized * (size_range[1] - size_range[0])

# Add normalized size columns to main dataframe
tutu_lc10a_aotu_syn_aggr['norm_size'] = tutu_lc10a_aotu_syn_aggr['count_post_aggr'].apply(
    lambda x: normalize_size(x, size_min, size_max)
)


fig, axn = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)

# Plot AOTU types separately with shared size mapping (no legends)
ax=axn[0]
aotu19_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']
sns.scatterplot(data=aotu19_data,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU19'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=False)  # No legend for individual plots
ax.set_title('TuTuA -> LC10a -> AOTU19\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu19, n_tutu_neurons, n_lc10a_aotu19, n_lc10a_neurons))

ax=axn[1]
aotu25_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']
sns.scatterplot(data=aotu25_data,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU25'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=False)  # No legend for individual plots
ax.set_title('TuTuA -> LC10a -> AOTU25\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu25, n_tutu_neurons, n_lc10a_aotu25, n_lc10a_neurons))

# Plot combined AOTU types with AOTU25 first (bottom layer), then AOTU19 (top layer)
# Only show TuTu neurons that are common to both AOTU types
ax=axn[2]


# Filter to only common TuTu neurons
tutu_common = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['bodyId_pre'].isin(common_tutus)]

# Plot AOTU25 first (bottom layer)
aotu25_combined = tutu_common[tutu_common['aotu_type'] == 'AOTU25']
sns.scatterplot(data=aotu25_combined,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU25'],
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.3,
                legend=False)

# Plot AOTU19 second (top layer)
aotu19_combined = tutu_common[tutu_common['aotu_type'] == 'AOTU19']
sns.scatterplot(data=aotu19_combined,
                ax=ax,
                x='y_post', y='z_post', 
                color=aotu_colors['AOTU19'],
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.3,
                legend=False)

# Create custom legends for both size and color (light gray)
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Calculate representative sizes
size_mid = (size_min + size_max) / 2
size_vals = [size_min, size_mid, size_max]
size_labels = [f'{int(size_min)}', f'{int(size_mid)}', f'{int(size_max)}']

# Create size legend elements (light gray)
size_legend_elements = []
for i, (val, label) in enumerate(zip(size_vals, size_labels)):
    # Calculate proportional size (20 to 300 pixels)
    prop_size = 20 + (val - size_min) / (size_max - size_min) * (300 - 20)
    size_legend_elements.append(
        Line2D([0], [0], marker='o', color='lightgray', linestyle='None',
               markersize=np.sqrt(prop_size)/2, label=label)
    )

# Create color legend elements (light gray)
color_legend_elements = [
    Line2D([0], [0], marker='o', color=aotu_colors['AOTU19'], linestyle='None',
           markersize=8, label='AOTU19'),
    Line2D([0], [0], marker='o', color=aotu_colors['AOTU25'], linestyle='None',
           markersize=8, label='AOTU25')
]

# Add size legend
size_legend = ax.legend(handles=size_legend_elements, 
                       title='Synapse Count', 
                       loc='upper left',
                       bbox_to_anchor=(1.05, 1))

# Add color legend
color_legend = ax.legend(handles=color_legend_elements,
                        title='AOTU Type',
                        loc='upper left', 
                        bbox_to_anchor=(1.05, 0.7))
ax.add_artist(size_legend)  # Keep size legend when adding color legend

ax.set_title(f'Common TuTu Neurons (n={len(tutu_common)})')

for ax in axn:
    ax.set_aspect('equal')
    ax.invert_yaxis()

#%%

# Color-code EACH TuTu neuron type by its LC10a->AOTU connections
tutu_types = tutu_lc10a_aotu_syn_aggr['bodyId_pre'].unique()
tutu_colors = sns.color_palette('colorblind', n_colors=len(tutu_types)).as_hex()[:]
tutu_cdict = {k: v for k, v in zip(tutu_types, tutu_colors)} 

fig, axn = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)

# Plot AOTU types separately with shared size mapping (no legends)
ax=axn[0]
aotu19_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']
# Add normalized size column
aotu19_data = aotu19_data.copy()
aotu19_data['norm_size'] = aotu19_data['count_post_aggr'].apply(
    lambda x: normalize_size(x, size_min, size_max)
)
sns.scatterplot(data=aotu19_data,
                ax=ax,
                x='y_post', y='z_post', 
                hue = 'bodyId_pre',
                palette=tutu_cdict,
                #color=aotu_colors['AOTU19'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=False)  # No legend for individual plots
ax.set_title('TuTuA -> LC10a -> AOTU19\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu19, n_tutu_neurons, n_lc10a_aotu19, n_lc10a_neurons))

ax=axn[1]
aotu25_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']
# Add normalized size column
aotu25_data = aotu25_data.copy()
aotu25_data['norm_size'] = aotu25_data['count_post_aggr'].apply(
    lambda x: normalize_size(x, size_min, size_max)
)
sns.scatterplot(data=aotu25_data,
                ax=ax,
                x='y_post', y='z_post', 
                hue = 'bodyId_pre',
                palette=tutu_cdict,
                #color=aotu_colors['AOTU25'], 
                size='norm_size', 
                sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=True)  # No legend for individual plots
ax.set_title('TuTuA -> LC10a -> AOTU25\n(n={}/{} TuTu, {}/{} LC10a)'.format(n_tutu_aotu25, n_tutu_neurons, n_lc10a_aotu25, n_lc10a_neurons))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), frameon=False)

for ax in axn:
    ax.set_aspect('equal')
ax.invert_yaxis()

#%% 
# Separately plot by each TuTu neuron 

fig, axn = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)

for i, (tutu_id, tutu_data) in enumerate(tutu_lc10a_aotu_syn_aggr.groupby('bodyId_pre')):
    ax=axn[i]
    sns.scatterplot(data=tutu_data,
                ax=ax,
                x='y_post', y='z_post', 
                hue = 'bodyId_pre',
                palette=tutu_cdict,
                #color=aotu_colors['AOTU25'], 
                #size='norm_size', 
                #sizes=(20, 300),  # Shared size range
                alpha=0.7,
                legend=i==n_tutu_neurons-1)  # No legend for individual plots

    ax.set_title(f'{tutu_id}')
    ax.set_aspect('equal')

ax.invert_yaxis()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), frameon=False)

#%%
import matplotlib as mpl

vmin = tutu_lc10a_aotu_syn_aggr['count_post_aggr'].min()
vmax = tutu_lc10a_aotu_syn_aggr['count_post_aggr'].max()
hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Plot TuTuA synapses to AOTU19 and AOTU25
fig, axn = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

# Create a consistent colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=hue_norm)
sm.set_array([])

for ai, (aotu_type, tutu_) in enumerate(tutu_lc10a_aotu_syn_aggr.groupby('aotu_type')):
    ax=axn[ai]
    sns.scatterplot(data=tutu_,
                ax=ax,
                x='y_post', y='z_post', 
                hue='count_post_aggr', 
                palette='viridis', hue_norm=hue_norm,
                size='norm_size', 
                sizes=(20, 300))  # Shared size range)
    ax.set_title('TuTuA -> LC10a -> {}'.format(aotu_type))
    
    # Remove individual legends
    ax.get_legend().remove()

ax.invert_yaxis()

for ax in axn:
    ax.set_aspect('equal')

# Add a single colorbar for both plots
cbar = fig.colorbar(sm, ax=axn, shrink=0.8, aspect=20)
cbar.set_label('Aggregated Count', rotation=270, labelpad=15)


#%%
# BOKEH PLOTTING 
# ------------------------------------------------------------

import bokeh
import bokeh.palettes
from bokeh.plotting import figure, show, output_notebook

#%a
sortvar = 'y'
sorted_lc10a_locs = lc10a_locs[lc10a_locs['type']=='dendrite'].sort_values(by=sortvar).copy()
sorted_lc10a_cell_ids = sorted_lc10a_locs['cell'] #[0::2]
clist = sns.color_palette('viridis', n_colors=len(sorted_lc10a_cell_ids)).as_hex()[:]
# colors = bokeh.palettes.cividis[len(cell_ids)] #[i]

# Add Bokeh-specific size column (scaled down from matplotlib range)
tutu_lc10a_aotu_syn_aggr['size_norm_bokeh'] = tutu_lc10a_aotu_syn_aggr['norm_size'].apply(
    lambda x: (x - 20) / (300 - 20) * (15 - 3) + 3
)

#%%
# Download some skeletons as DataFrames and attach columns for bodyId and color
skeletons = []
for i, bodyId in enumerate(sorted_lc10a_cell_ids):
    s = neu.fetch_skeleton(bodyId, format='pandas')
    s['bodyId'] = bodyId
    s['color'] = clist[i] #bokeh.palettes.Accent[5][i]
    skeletons.append(s)

# Combine into one big table for convenient processing
skeletons = pd.concat(skeletons, ignore_index=True)
skeletons.head()

# Join parent/child nodes for plotting as line segments below.
# (Using each row's 'link' (parent) ID, find the row with matching rowId.)
segments = skeletons.merge(skeletons, 'inner',
                           left_on=['bodyId', 'link'],
                           right_on=['bodyId', 'rowId'],
                           suffixes=['_child', '_parent'])
#%%
p = figure()
p.y_range.flipped = True
p.background_fill_color = None
p.border_fill_color = None

set_bokeh_line_colors(p, c="lightgrey")
# p.background_fill_alpha = 0
#ax = fig.add_subplot(projection='3d')
# Plot skeleton segments (in 2D)
# p.segment(x0='x_child', x1='x_parent',
#           y0='z_child', y1='z_parent',
#           color='color_child',
#           source=segments)
p.segment(x0='{}_child'.format(sortvar), x1='{}_parent'.format(sortvar),
          y0='z_child', y1='z_parent',
          color='color_child',
          source=segments)

for i, cell in enumerate(sorted_lc10a_cell_ids):
    df_ = sorted_lc10a_locs[sorted_lc10a_locs['cell']==cell]
    p.scatter(df_[sortvar], df_['z'], color=clist[i], marker='o')
p.xaxis.axis_label = sortvar
p.yaxis.axis_label = 'z'

# %

# Create 3 Bokeh subplots in one figure
from bokeh.layouts import row as bokeh_row, column, gridplot
from bokeh.models import HoverTool

# Create color normalization for count_post_aggr (for Bokeh color mapping)
from bokeh.palettes import viridis
bokeh_colors = viridis(256)  # High resolution color palette

def get_bokeh_color_for_count(count):
    """Get color based on global count range for Bokeh"""
    if size_max == size_min:
        return bokeh_colors[0]
    norm_count = (count - size_min) / (size_max - size_min)
    color_idx = int(norm_count * (len(bokeh_colors) - 1))
    return bokeh_colors[color_idx]

# Ensure size_norm_bokeh column exists for Bokeh plotting
if 'size_norm_bokeh' not in tutu_lc10a_aotu_syn_aggr.columns:
    tutu_lc10a_aotu_syn_aggr['size_norm_bokeh'] = tutu_lc10a_aotu_syn_aggr['norm_size'].apply(
        lambda x: (x - 20) / (300 - 20) * (15 - 3) + 3
    )

# Create 3 separate figures
p1 = figure(title="LC10a Skeleton Segments", width=300, height=300)
p1.y_range.flipped = True
p1.background_fill_color = None
p1.border_fill_color = None
set_bokeh_line_colors(p1, c="lightgrey")

p2 = figure(title="TuTuA → LC10a → AOTU19", width=300, height=300)
p2.y_range.flipped = True
set_bokeh_line_colors(p2, c="lightgrey")

p3 = figure(title="TuTuA → LC10a → AOTU25", width=300, height=300)
p3.y_range.flipped = True
set_bokeh_line_colors(p3, c="lightgrey")

# Plot 1: LC10a skeleton segments
# Plot skeleton segments (in 2D)
p1.segment(x0=f'{sortvar}_child', x1=f'{sortvar}_parent',
           y0='z_child', y1='z_parent',
           color='color_child',
           source=segments)

# Add scatter points for each cell
for i, cell in enumerate(sorted_lc10a_cell_ids):
    df_ = sorted_lc10a_locs[sorted_lc10a_locs['cell']==cell]
    p1.scatter(df_[sortvar], df_['z'], color=clist[i], marker='o', size=4)

# Plot 2: AOTU19 synapses using tutu_lc10a_aotu_syn_aggr data
aotu19_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU19']
if len(aotu19_data) > 0:
    for i, (idx, row) in enumerate(aotu19_data.iterrows()):
        color = get_bokeh_color_for_count(row['count_post_aggr'])  # Color by count
        size = row['size_norm_bokeh']  # Use the Bokeh-specific size column
        p2.scatter(row['y_post'], row['z_post'], 
                   color=color, alpha=0.7, size=size)

# Plot 3: AOTU25 synapses using tutu_lc10a_aotu_syn_aggr data
aotu25_data = tutu_lc10a_aotu_syn_aggr[tutu_lc10a_aotu_syn_aggr['aotu_type'] == 'AOTU25']
if len(aotu25_data) > 0:
    for i, (idx, row) in enumerate(aotu25_data.iterrows()):
        color = get_bokeh_color_for_count(row['count_post_aggr'])  # Color by count
        size = row['size_norm_bokeh']  # Use the Bokeh-specific size column
        p3.scatter(row['y_post'], row['z_post'], 
                   color=color, alpha=0.7, size=size)

# Set axis labels
p1.xaxis.axis_label = 'y'
p1.yaxis.axis_label = 'z'
p2.xaxis.axis_label = 'y'
p2.yaxis.axis_label = 'z'
p3.xaxis.axis_label = 'y'
p3.yaxis.axis_label = 'z'

# Set axis limits for all plots to match the synapse data range
if len(tutu_lc10a_aotu_syn_aggr) > 0:
    # Use the aggregated data to get overall range
    y_range = [tutu_lc10a_aotu_syn_aggr['y_post'].min() - 1000, tutu_lc10a_aotu_syn_aggr['y_post'].max() + 1000]
    z_range = [tutu_lc10a_aotu_syn_aggr['z_post'].min() - 1000, tutu_lc10a_aotu_syn_aggr['z_post'].max() + 1000]
    
    # Set the same range for all plots
    # For y-axis: set start to high value, end to low value (big to small)
    p1.x_range.start = y_range[0]
    p1.x_range.end = y_range[1]
    p1.y_range.start = z_range[1]  # High z value at top
    p1.y_range.end = z_range[0]    # Low z value at bottom
    
    p2.x_range.start = y_range[0]
    p2.x_range.end = y_range[1]
    p2.y_range.start = z_range[1]  # High z value at top
    p2.y_range.end = z_range[0]    # Low z value at bottom
    
    p3.x_range.start = y_range[0]
    p3.x_range.end = y_range[1]
    p3.y_range.start = z_range[1]  # High z value at top
    p3.y_range.end = z_range[0]    # Low z value at bottom

# Synchronize axes across all plots using Bokeh's range linking
p2.x_range = p1.x_range
p2.y_range = p1.y_range
p3.x_range = p1.x_range
p3.y_range = p1.y_range

# Create layout with 3 subplots
layout = bokeh_row(p1, p2, p3)

# Show the combined plot
show(layout)

# %%
