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
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# import neuprint stuff
import neuprint as neu
from neuprint import NeuronCriteria as NC

import bokeh.palettes
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import export_png
output_notebook()

# colorbar for hue_var and palette - create mappable from categorical palette
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# 
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
# Get all TuTuA_2 neurons
# ======================================================
TuTuA2_neurons, TuTuA2_roi_counts = neu.fetch_neurons(NC(type='TuTuA_2', 
                                                         client=c))
TuTuA2_neurons.head()

#%%
# Get all LC10a neurons
LC10a_neurons, LC10a_roi_counts = neu.fetch_neurons(NC(type='LC10a', 
                                                         client=c))
LC10a_neurons.head()


#%%
# Get AOTU019 and AOTU025 neurons for reference
side = 'R'

aotu19_neurons, aotu19_roi_counts = neu.fetch_neurons(NC(type='AOTU019', 
                                                         client=c))
aotu25_neurons, aotu25_roi_counts = neu.fetch_neurons(NC(type='AOTU025', 
                                                         client=c))

aotu19_ids = aotu19_neurons[aotu19_neurons['instance']==f'AOTU019_{side}']['bodyId'].unique()
aotu25_ids = aotu25_neurons[aotu25_neurons['instance']==f'AOTU025_{side}']['bodyId'].unique()
print(f"Number of AOTU019-{side} neurons: {len(aotu19_ids)}")
print(f"Number of AOTU025-{side} neurons: {len(aotu25_ids)}")

lc10a_ids = LC10a_neurons[LC10a_neurons['instance']==f'LC10a_{side}']['bodyId'].unique()
print(f"Number of LC10a-{side} neurons: {len(lc10a_ids)}")

#%%
# Get LC10a synapse
lc10a_syn = neu.fetch_synapses(lc10a_ids, client=c)
#%
#lc10a_syn = lc10a_syn[lc10a_syn['roi_post']==f'AOTU({side})']
print(f"Number of LC10a synapses: {len(lc10a_syn)}")
lc10a_syn.head()

#%%
# Get LC10a -> AOTU19/25 synapses
lc10a_aotu19_syn = neu.fetch_synapse_connections(lc10a_ids, aotu19_ids, client=c)
lc10a_aotu25_syn = neu.fetch_synapse_connections(lc10a_ids, aotu25_ids, client=c)
print(f"Number of LC10a -> AOTU19 synapses: {len(lc10a_aotu19_syn)}")
print(f"Number of LC10a -> AOTU25 synapses: {len(lc10a_aotu25_syn)}")

#%%
x = 'z'
y = 'y'
# Plot AOTU19 synapses and AOTU025 synapses
fig, ax = plt.subplots( figsize=(5, 5))
ax.set_title(f'LC10a-{side} synapses')
# also plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=f'{x}', y=f'{y}', color='lightgray',
                s=5)

# plot aotu019 and 025
sns.scatterplot(data=lc10a_aotu19_syn, ax=ax,
                x=f'{x}_post', y=f'{y}_post', s=5, alpha=0.1,
                color='red', label='AOTU19')
sns.scatterplot(data=lc10a_aotu25_syn, ax=ax,
                x=f'{x}_post', y=f'{y}_post', s=5, alpha=0.1,
                color='blue', label='AOTU25')
ax.legend(frameon=False)
ax.set_aspect('equal')

ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'{side}_LC10a-synapses_on_AOTU19-25'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)


#%%

LO_coords = lc10a_syn[(lc10a_syn['bodyId'].isin(lc10a_ids)
                    & (lc10a_syn['roi']==f'LO({side})'))].copy()

sort_by = 'y'
# Sort bodyids by sory_by
sorted_lc10a_ids = LO_coords.sort_values(by=sort_by)['bodyId'].unique()

#%%
# Visulize L10a neurons to orient ourselves
# ======================================================    

# Create skeletons WITHOUT colors (so we can change colormapping later)
skeletons = []
for i, bodyId in enumerate(sorted_lc10a_ids):
    s = neu.fetch_skeleton(bodyId, format='pandas')
    s['bodyId'] = bodyId
    skeletons.append(s) 

skeletons = pd.concat(skeletons, ignore_index=True)
skeletons.head()

#%%
# Join parent/child nodes for plotting as line segments below.
# (Using each row's 'link' (parent) ID, find the row with matching rowId.)
segments = skeletons.merge(skeletons, 'inner',
                           left_on=['bodyId', 'link'],
                           right_on=['bodyId', 'rowId'],
                           suffixes=['_child', '_parent'])

#%%
# Simple function to set colors - call this with different palette names to change colors
def set_segment_colors(segments, body_ids, palette='Viridis', n_colors=256):
    """Update color columns in segments DataFrame based on palette."""
    palette_func = getattr(bokeh.palettes, palette, bokeh.palettes.Viridis)
    colors = palette_func[n_colors] if n_colors in palette_func else palette_func[256]
    
    color_map = {bodyId: colors[int((i / len(body_ids)) * (len(colors) - 1))] 
                for i, bodyId in enumerate(body_ids)}
    
    # bodyId doesn't get a suffix because it's used as a merge key
    # Both child and parent have the same bodyId in each row
    segments['color_child'] = segments['bodyId'].map(color_map)
    segments['color_parent'] = segments['bodyId'].map(color_map)
    
    return segments

#%%
# Plot LC10a skeleton and synapses
# ======================================================
plot_aotu_syn = True
aotu_str = '_on_AOTU19-25' if plot_aotu_syn else ''

# Apply initial colors
set_segment_colors(segments, sorted_lc10a_ids, palette='Viridis')

# Plot skeletons
p = figure()
p.y_range.flipped = True

# Plot skeleton segments (in 2D)
seg_renderer = p.segment(x0=f'{x}_child', x1=f'{x}_parent',
                         y0=f'{y}_child', y1=f'{y}_parent',
                         color='color_child',
                         source=segments)

# To change colors later, just run these two lines:
# set_segment_colors(segments, sorted_lc10a_ids, palette='Cividis')  # Try: 'Viridis', 'Cividis', 'Plasma', 'Inferno'
# seg_renderer.data_source.data = {col: segments[col].values for col in segments.columns}
# label axes
p.xaxis.axis_label = x
p.yaxis.axis_label = y

if plot_aotu_syn:
    # Also plot the synapses from the above example
    # p.scatter(points['x_post'], points['z_post'], color=points['color'])
    p.scatter(lc10a_aotu19_syn[f'{x}_post'], 
            lc10a_aotu19_syn[f'{y}_post'], 
            color='red', size=2, alpha=0.7)
    p.scatter(lc10a_aotu25_syn[f'{x}_post'], 
            lc10a_aotu25_syn[f'{y}_post'], 
            color='blue', size=2, alpha=0.7) 

# title axis
p.title.text = f'LC10a-{side} synapses'


# Save as HTML only (avoid document ownership/show/export issues)
#figname = f'skel_{side}_LC10a-synapses{aotu_str}'
#p.save(filename=os.path.join(output_dir, f'{figname}.png'))
#print(figname)

show(p)


# %%
# Get synapse connections from TuTuA_2 to LC10a neurons
src = NC(type='TuTuA_2')
dst = NC(type='LC10a')

# bodyId_pre are the TuTuA_2 neurons
# bodyId_post are the LC10a neurons
tutu_lc10a_syn = neu.fetch_synapse_connections(src, dst, client=c,
                                               nt='max',
                                               min_total_weight=10)
# %%

tutu_lc10a_syn_side = tutu_lc10a_syn[tutu_lc10a_syn['bodyId_post'].isin(lc10a_ids)].copy().reset_index(drop=True)
print(f"Number of synapses to LC10a-{side} neurons: {tutu_lc10a_syn_side['bodyId_post'].nunique()}")

# add order from sorted_lc10a_ids
tutu_lc10a_syn_side[f'{sort_by}_order'] = tutu_lc10a_syn_side['bodyId_post'].map(dict(zip(sorted_lc10a_ids, range(len(sorted_lc10a_ids)))))
tutu_lc10a_syn_side.head()

# Add synapse counts per LC10a neuron
tutu_lc10a_syn_side['syn_count'] = tutu_lc10a_syn_side.groupby(['bodyId_post'])['bodyId_post'].transform('count')

#%%
# Plot synapse locations from TuTuA_2 to onto LC10a (And AOTU019/25 as reference)
# -------------------------------------------------------------
x = 'z'
y = 'y'
#z = 'z'
pre_post = 'pre'
xvar = f'{x}_{pre_post}'
yvar = f'{y}_{pre_post}'
huevar = f'{sort_by}_order' #= f'{z}_{pre_post}'

fig, axn = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax=axn[0]
# plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y,
                #x=f'{x}', y=f'{y}', 
                color='lightgray', s=5, alpha=0.5)
# Plot TuTuA synapses
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar, y=yvar, hue=huevar,
                palette='viridis', legend=0, 
                s=5)
title = f'TuTuA2 {pre_post}, hue=sorted LC10a {sort_by}-pos in LO'
ax.set_title(title, loc='left', fontsize=12) 

ax=axn[1]
# plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y,
                #x=f'{x}', y=f'{y}', 
                color='lightgray', s=5, alpha=0.5)
# Plot AOTU019 and 25 
sns.scatterplot(data=lc10a_aotu19_syn, ax=ax,
                x=xvar, y=yvar, color='red', s=5, alpha=0.5, label='AOTU019')
sns.scatterplot(data=lc10a_aotu25_syn, ax=ax,
                x=xvar, y=yvar, color='blue', s=5, alpha=0.5, label='AOTU025')
ax.legend(frameon=False)
ax.set_title('AOTU019 and 25', loc='left', fontsize=12)
#ax.set_title(f'hue=syn_count by LC10a neuron', loc='left', fontsize=12)


ax.set_aspect('equal')
ax.invert_yaxis()

# save
putil.label_figure(fig, figid)
figname = f'{side}_{pre_post}-synaptic_on_LC10a_huesort-{sort_by}-pos-in-LO'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

# %%
# Zoom in 

fig, axn = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
# also plot L1a0 neurons
ax=axn[0]
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y,
                color='lightgray', s=5, alpha=0.5)
# Plot TuTuA synapses
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar, y=yvar, hue=huevar,
                palette='viridis', legend=0, 
                s=20)
fig.suptitle(f'TuTuA2->LC10a, {pre_post}-synaptic')
ax.set_title(f'hue=sort by LC10a {sort_by}-pos in LO', loc='left', fontsize=12)

# Plot by synapse
ax=axn[1]
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y,
                color='lightgray', s=5, alpha=0.5)
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar, y=yvar, hue='syn_count',
                palette='viridis', legend=0, 
                s=20)
ax.set_title(f'hue=syn_count by LC10a neuron', loc='left', fontsize=12)

for ax in axn:
    ax.set_aspect('equal')
    ax.set_xlim(15000, 22000)
    ax.set_ylim([12000, 20000])
    ax.set_aspect('equal')
    ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'{side}_{pre_post}-synaptic_on_LC10a_ZOOM_hue-{sort_by}pos-and-syncount'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
xvar = 'z_pre'
yvar = 'y_pre'
g = sns.jointplot(data=tutu_lc10a_syn_side, x=xvar, y=yvar, kind='hist') #, bins=20)
g.ax_joint.set_aspect('equal')
g.ax_joint.invert_yaxis()

#%%
from sklearn.decomposition import PCA

def do_pca_on_synapses(tutu_lc10a_syn, xvar='post_z', yvar='post_y', verbose=False):
    # Do PCA to find the top 2 PCs of the synapse coordinates in z, y space
    # Combine Z and Y coordinates into a 2D array
    zy_coords = tutu_lc10a_syn[[xvar, yvar]].copy().values

    lc10a_pca = PCA(n_components=2)
    lc10a_pca.fit(zy_coords)
    lc10a_pca_scores = lc10a_pca.transform(zy_coords)
    lc10a_pca_scores = pd.DataFrame(lc10a_pca_scores, columns=['PC1', 'PC2'])

    if verbose:
        print(f"PCA explained variance ratio: {lc10a_pca.explained_variance_ratio_}")
        print(f"Total explained variance: {lc10a_pca.explained_variance_ratio_.sum():.3f}")

    return lc10a_pca_scores


def plot_pca_transformed(tutu_lc10a_syn, tutu_lc10a_syn_pca, lc10a_cdict,
                         xvar='post_z', yvar='post_y', hue_var='post_root_id',
                         markersize=20, marker='x', invert_yaxis=True):
    # Plot to check
    fig, axn = plt.subplots(1, 2, figsize=(10, 5))
    ax=axn[0]
    # Plot original data
    sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                    x=xvar, y=yvar, 
                    hue=hue_var, palette=lc10a_cdict, legend=0,
                    marker=marker, s=markersize)
    ax.set_title('Original', fontsize=8, loc='left')
    ax.set_aspect('equal')
    if invert_yaxis:
        ax.invert_yaxis()
    #ax.invert_xaxis()

    ax=axn[1]
    # Plot PCA scores
    sns.scatterplot(data=tutu_lc10a_syn_pca, ax=ax,
                    x='PC1', y='PC2',
                    hue=hue_var, palette=lc10a_cdict, legend=0,
                    marker=marker, s=markersize)
                    ##hue='syn_count', palette='magma', legend=1)
    ax.set_aspect('equal')
    ax.set_title('PCA transformed', fontsize=8, loc='left')

    if invert_yaxis:
        ax.invert_yaxis()

    plt.subplots_adjust(wspace=0.5)

    return fig

def bin_pca_scores(tutu_lc10a_syn_pca, n_bins=20):
    # Bin PC1 axis into even bins, label them by the middle of the bin
    #n_bins = 20
    # 
    bins = np.linspace(tutu_lc10a_syn_pca['PC1'].min(), tutu_lc10a_syn_pca['PC1'].max(), n_bins)
    tutu_lc10a_syn_pca['PC1_bin'] = pd.cut(tutu_lc10a_syn_pca['PC1'], bins)
    tutu_lc10a_syn_pca['PC1_bin_label'] = tutu_lc10a_syn_pca['PC1_bin'].apply(lambda x: x.mid)

    # Do the same for PC2
    n_bins_pc2 = n_bins
    bins_pc2 = np.linspace(tutu_lc10a_syn_pca['PC2'].min(), 
                           tutu_lc10a_syn_pca['PC2'].max(), n_bins_pc2)
    tutu_lc10a_syn_pca['PC2_bin'] = pd.cut(tutu_lc10a_syn_pca['PC2'], bins_pc2)
    tutu_lc10a_syn_pca['PC2_bin_label'] = tutu_lc10a_syn_pca['PC2_bin'].apply(lambda x: x.mid)

    # Count how many points fall into each bin
    # Should be the same as summing each neuron's synapses in that bin
    tutu_lc10a_syn_pca_binned1= tutu_lc10a_syn_pca.groupby(['PC1_bin_label'])\
                                    .agg({'PC1': 'count', 'syn_count': 'sum'}).reset_index()
                                    #.agg({'PC1': 'count'}).reset_index()

    # Same to count PC2
    tutu_lc10a_syn_pca_binned2 = tutu_lc10a_syn_pca.groupby(['PC2_bin_label'])\
                                    .agg({'PC2': 'count', 'syn_count': 'sum'}).reset_index()   
                                    #.agg({'PC2': 'count'}).reset_index()   
    tutu_lc10a_syn_pca_binned = pd.concat([tutu_lc10a_syn_pca_binned1, 
                                           tutu_lc10a_syn_pca_binned2], axis=1)
    # Combine counts for PC1 and 2

    # Convert PC1_bin_label to numeric to ensure proper x-axis alignment
    tutu_lc10a_syn_pca_binned['PC1_bin_numeric'] = pd.to_numeric(tutu_lc10a_syn_pca_binned['PC1_bin_label'])
    tutu_lc10a_syn_pca_binned['PC2_bin_numeric'] = pd.to_numeric(tutu_lc10a_syn_pca_binned['PC2_bin_label'])

    return tutu_lc10a_syn_pca_binned

def plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, lc10a_cdict,
                          hue_var='post_root_id', bin_cmap = 'viridis_r',
                          markersize=20, marker='x', alpha=1,
                          marginal_marker='o', marginal_markersize=20):
    #bin_cmap = 'viridis_r'

    # Plot the data in the new basis with PC1 and PC2
    fig, axn = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
    # Reduce white space by adjusting subplot layout more aggressively
    plt.subplots_adjust(top=0.98, bottom=0.12, hspace=0.3)

    # PC1 distribution (top subplot)
    ax_pc1 = axn[0]
    sns.scatterplot(data=tutu_lc10a_syn_pca_binned, 
                x='PC1_bin_numeric', y='PC1',
                hue='PC1_bin_label',
                palette=bin_cmap, legend=0, ax=ax_pc1,
                marker=marginal_marker, s=marginal_markersize)
    ax_pc1.set_xlabel('PC1')
    ax_pc1.set_ylabel('Count')
    #ax_pc1.set_title('PC1 Distribution'
    #
    # Main scatter plot (bottom subplot)
    ax_main = axn[1]
    sns.scatterplot(data=tutu_lc10a_syn_pca, ax=ax_main,
                    x='PC1', y='PC2',
                    hue=hue_var, marker=marker, s=markersize,
                    palette=lc10a_cdict, legend=0, alpha=alpha)
    ax_main.set_aspect('equal')

    # Get the x-axis limits from the main plot after it's been drawn
    x_min, x_max = ax_main.get_xlim()

    # Set the exact same x-axis limits as the main plot
    ax_pc1.set_xlim(x_min, x_max)

    # Force both subplots to have the same width by adjusting their positions
    # Get the position of the main subplot
    pos_main = ax_main.get_position()
    pos_pc1 = ax_pc1.get_position()

    # Make the top subplot shorter and positioned right above the main subplot
    # Calculate new position: same x, just above main plot, same width, shorter height
    new_y0 = pos_main.y1 + 0.02  # Position it just above the main plot
    new_height = 0.1  # Make it much shorter
    ax_pc1.set_position([pos_main.x0, new_y0, pos_main.width, new_height])

    # Add third subplot for PC2 binned data
    # Get the position of the PC1 subplot (top) - after repositioning
    pos_pc1 = ax_pc1.get_position()
    # Get the position of the main subplot (bottom)
    pos_main = ax_main.get_position()

    # Create third subplot with:
    # - Height matching main plot (pos_main.height)
    # - Width matching PC1 subplot height (pos_pc1.height) - PHYSICAL SIZE
    # - Positioned to the right of the main plot
    ax_pc2 = fig.add_axes([pos_main.x1 + 0.01, pos_main.y0, pos_pc1.height-0.04, pos_main.height])

    # Plot PC2 binned data as horizontal bar chart
    sns.scatterplot(data=tutu_lc10a_syn_pca_binned, 
                y='PC2_bin_numeric', x='PC2',
                hue='PC2_bin_label',
                palette=bin_cmap, legend=0, ax=ax_pc2,
                marker=marginal_marker, s=marginal_markersize)
    ax_pc2.set_yticklabels([])
    ax_pc2.set_ylabel('')
    #ax_pc2.set_yticklabels([])
    ax_pc1.set_xticklabels([])

    #plt.subplots_adjust(top=0.8)
    for ax in [ax_pc1, ax_main, ax_pc2]:
        sns.despine(ax=ax, top=True, right=True)

    return fig


#%%
# PCA on synapses (TuTu->LC10a)
# =======================================================
# Filter to only synapses to sorted LC10a neurons
#tutu_lc10a_syn_side = tutu_lc10a_syn[tutu_lc10a_syn['bodyId_post'].isin(sorted_lc10a_ids)].copy().reset_index(drop=True)


#%%
# Create dictionary of colors
hue_palette = 'viridis'
sorted_lc10a_ids_list = list(sorted_lc10a_ids)
lc10a_colors = sns.color_palette(hue_palette, n_colors=len(sorted_lc10a_ids_list))
lc10a_cdict = dict(zip(sorted_lc10a_ids_list, lc10a_colors)) 

# Create a continuous mappable based on the order in sorted_lc10a_ids
lc10a_listed_cmap = ListedColormap(lc10a_colors)

#%%
fig, axn = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
x = 'z'
y = 'y'
ax=axn[0]
# plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y, hue='bodyId', 
                palette=lc10a_cdict, legend=0,
                #x=f'{x}', y=f'{y}', 
                s=5, alpha=0.5)
ax.invert_yaxis()
sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label('LC10a LO position (sorted order)')

ax=axn[1]
# plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=x, y=y,
                #x=f'{x}', y=f'{y}', 
                color='lightgray', s=5, alpha=0.5)
ax.set_title('LC10a neurons', fontsize=12, loc='left')

# Plot TuTuA synapses
scatter = sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=f'{x}_pre', y=f'{y}_pre', 
                hue='syn_count', palette='viridis', legend=0, alpha=0.5,
                size=5) #'syn_count', alpha=0.5, s=5)
ax.set_title('TuTuA synapses', fontsize=12, loc='left')

# colorbar for hue_var and palette - create proper mappable for continuous data
import matplotlib.cm as cm
norm = plt.Normalize(vmin=tutu_lc10a_syn_side['syn_count'].min(), 
                     vmax=tutu_lc10a_syn_side['syn_count'].max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label('Synapse Count')

for ax in axn:
    ax.set_xlim(15000, 22000)
    ax.set_ylim([13000, 20000])
    ax.set_aspect('equal')
    ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'{side}_LC10a_position_TuTuA2_syncount'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%

# Convert coords
lc10a_pca_scores = do_pca_on_synapses(tutu_lc10a_syn_side, xvar='z_post', yvar='y_post')

#% # Add PCA scores to the original dataframe
tutu_lc10a_syn_pca = pd.concat([tutu_lc10a_syn_side, lc10a_pca_scores], axis=1)

fig = plot_pca_transformed(tutu_lc10a_syn_side, tutu_lc10a_syn_pca, lc10a_cdict,
                           xvar='z_post', yvar='y_post', hue_var='bodyId_post',
                           marker='o' )
                         
putil.label_figure(fig, figid)
figname = 'check_pca_tutu-lc10a'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# Bin PCA scores and plot
tutu_lc10a_syn_pca_binned = bin_pca_scores(tutu_lc10a_syn_pca)
#%
#%
bin_cmap = 'viridis_r'
fig = plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, 
                            lc10a_cdict, bin_cmap=bin_cmap,
                            markersize=10, marker='o',
                            hue_var='bodyId_post', 
                            marginal_marker='o', marginal_markersize=20)
putil.label_figure(fig, figid)
figname = f'{side}_pca_scores_binned'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)


# %%
#tutu_lc10a_syn_slice

# FIND THE CORRECT SLICING TO MATCH 2p - LC10a terminals
# -------------------------------------
xvar = 'x'
yvar = 'y'
zvar = 'z'

markersize = 10
alpha = 1

plotd = lc10a_syn[lc10a_syn['roi']==f'AOTU({side})']

zmin, zmax = plotd[f'{zvar}'].min(), plotd[f'{zvar}'].max()
print( f'z range: {zmin} to {zmax}')
z_depth = zmax - zmin

z_steps = 5

fig, axn = plt.subplots(1, z_steps, figsize=(4*z_steps, 6),
                        sharex=True, sharey=True)
ri = 0
for i in range(z_steps):
    curr_z_slice = [zmin + i*z_depth/z_steps, zmin + (i+1)*z_depth/z_steps]
    print(curr_z_slice)
    lc10a_syn_slice = plotd[plotd[f'{zvar}'].between(curr_z_slice[0], curr_z_slice[1])]
    ax=axn[i]
    # add lc10a 
    sns.scatterplot(data=lc10a_syn, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    color='lightgray',
                    s=markersize, edgecolor='none', alpha=alpha)

    sns.scatterplot(data=lc10a_syn_slice, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    hue='bodyId', palette=lc10a_cdict, legend=0,
                    s=markersize, edgecolor='none', alpha=alpha)
    
    ax.set_title(f'{zvar} slice: {curr_z_slice[0]} to {curr_z_slice[1]}')
# zoom in
lc10a_syn_slice_xlim = plotd[f'{xvar}'].min(), plotd[f'{xvar}'].max()
lc10a_syn_slice_ylim = plotd[f'{yvar}'].min(), plotd[f'{yvar}'].max()
ax.set_xlim(lc10a_syn_slice_xlim)
ax.set_ylim(lc10a_syn_slice_ylim)

for ax in axn:
    ax.set_aspect('equal')
ax.invert_yaxis()

#%%

fig, axn = plt.subplots(2, z_steps, figsize=(4*z_steps, 6),
                        sharex=True, sharey=True)
ri = 0
for i in range(z_steps):
    curr_z_slice = [zmin + i*z_depth/z_steps, zmin + (i+1)*z_depth/z_steps]
    print(curr_z_slice)
    lc10a_syn_slice = plotd[plotd[f'{zvar}'].between(curr_z_slice[0], curr_z_slice[1])]
    ax=axn[ri, i]
    sns.scatterplot(data=lc10a_syn_slice, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    hue='bodyId', palette=lc10a_cdict, legend=0,
                    s=markersize, edgecolor='none', alpha=alpha)
    ax.set_title(f'{zvar} slice: {curr_z_slice[0]} to {curr_z_slice[1]}')

# Add colorbar shared only for TOP row of subplots:
cbar_ax = fig.add_axes([0.91, 0.6, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('LO position (sorted order)')

#%
# Check TuTuA synapses at a given z slice
#z_steps = 5
#fig, axn = plt.subplots(1, z_steps, figsize=(5*z_steps, 5),
#                        sharex=True, sharey=True)
ri = 1
for i in range(z_steps):
    curr_z_slice = [zmin + i*z_depth/z_steps, zmin + (i+1)*z_depth/z_steps]
    print(curr_z_slice)
    tutu_lc10a_syn_slice = tutu_lc10a_syn_side[tutu_lc10a_syn_side[f'{zvar}_pre'].between(curr_z_slice[0], curr_z_slice[1])]
    ax=axn[ri, i]
    sns.scatterplot(data=tutu_lc10a_syn_slice, ax=ax,
                    x=f'{xvar}_pre', y=f'{yvar}_pre',
                    hue='syn_count', palette='viridis', legend=0,
                    size=markersize, edgecolor='none', alpha=alpha)
    ax.set_title(f'{zvar} slice: {curr_z_slice[0]} to {curr_z_slice[1]}')

# Add colorbar shared only for BOTTOM row of subplots:
cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.3])
sm = cm.ScalarMappable(cmap='viridis')
sm.set_clim(vmin=tutu_lc10a_syn_side['syn_count'].min(), 
            vmax=tutu_lc10a_syn_side['syn_count'].max())
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Synapse Count')

# axes
for ax in axn.flat:
    ax.set_aspect('equal')
ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'{side}_TuTuA_synapses_by_zslice'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# Do the binning by zvar -- count
# import mplt normalize
import matplotlib.colors as mcolors
# add bin column with the same bin slices (vectorized)
bins = np.linspace(zmin, zmax, z_steps + 1)
bin_midpoints = (bins[:-1] + bins[1:]) / 2
tutu_lc10a_syn_side['zvar_bin_label'] = pd.cut(
    tutu_lc10a_syn_side[f'{zvar}_pre'],
    bins=bins,
    labels=bin_midpoints,
    include_lowest=True
).astype(float)
    
    
# Group by zvar bin, and count synapses
count_col = 'bodyId_post' # doesnt matter which column we count
tutu_lc10a_syn_side_binned = tutu_lc10a_syn_side.groupby(['zvar_bin_label'])[count_col].count().reset_index()       

# Check if the syanpses actually match the bin
n_bins = tutu_lc10a_syn_side['zvar_bin_label'].nunique()
fig, axn = plt.subplots(1, n_bins, figsize=(5*n_bins, 5), sharex=True, sharey=True)

syn_count_range = tutu_lc10a_syn_side['syn_count'].min(), tutu_lc10a_syn_side['syn_count'].max()
vmin, vmax = syn_count_range
hue_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
for i, (v, tmp) in enumerate(tutu_lc10a_syn_side.groupby('zvar_bin_label')):
    ax = axn[i]
    sns.scatterplot(data=tmp, ax=ax,
                x=f'{xvar}_post', y=f'{yvar}_post',
                hue='syn_count', palette='viridis', legend=0, 
                hue_norm=hue_norm,
                s=20, alpha=1)
    ax.set_aspect('equal')
    curr_zrange = tmp[f'{zvar}_pre'].min(), tmp[f'{zvar}_pre'].max()
    ax.set_title(f'{zvar} slice: {curr_zrange[0]} to {curr_zrange[1]}')
ax.invert_yaxis()


#%%
# Count N synapses in each of these bins:

# Plot synapses by zvar bin
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=tutu_lc10a_syn_side_binned, ax=ax,
                x='zvar_bin_label', y='bodyId_post',
                palette='viridis', legend=0,
                s=20, alpha=1)
ax.set_xlabel(f'{zvar} slice')
#ax.set_aspect('equal')
ax.invert_yaxis()

print(tutu_lc10a_syn_side_binned)

#%%


# SCRATCH
# ==============================
#%%
# Bin synapses by LC10a LO position instead??
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Add sorting order to tutu_lc10a_syn_side
tutu_lc10a_syn_side['bodyId_post_order'] = tutu_lc10a_syn_side['bodyId_post'].map(dict(zip(sorted_lc10a_ids, range(len(sorted_lc10a_ids)))))

sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=f'{xvar}_pre', y=f'{yvar}_pre',
                hue='bodyId_post_order', palette='viridis', legend=0,
                s=20, alpha=1)
ax.set_aspect('equal')
ax.invert_yaxis()


#%%
# Bin neurons by their order
tutu_lc10a_syn_side['bodyId_bin'] = pd.cut(tutu_lc10a_syn_side['bodyId_post_order'], bins=len(sorted_lc10a_ids))
tutu_lc10a_syn_side['bodyId_bin_label'] = tutu_lc10a_syn_side['bodyId_bin'].apply(lambda x: x.mid)

# Group by bodyId bin, and count synapses
tutu_lc10a_syn_side_binned = tutu_lc10a_syn_side.groupby(['bodyId_bin_label'])['syn_count'].sum().reset_index()

# Plot synapses by bodyId bin
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=tutu_lc10a_syn_side_binned, ax=ax,
                x='bodyId_bin_label', y='syn_count',
                palette='viridis', legend=0,
                s=20, alpha=1)
#ax.set_aspect('equal')
#ax.invert_yaxis()



#%%


#%%
# Do PCA on this slice
z_start = 32487
z_end = 34067

curr_tutu_lc10a_syn_slice = tutu_lc10a_syn_side[tutu_lc10a_syn_side[f'{zvar}_pre'].between(z_start, z_end)].reset_index(drop=True)
tutu_lc10a_syn_slice_pca = do_pca_on_synapses(curr_tutu_lc10a_syn_slice, 
                                              xvar=f'{xvar}_pre', yvar=f'{yvar}_pre')

# Add PC scores to the original dataframe
tutu_lc10a_syn_slice_pca = pd.concat([curr_tutu_lc10a_syn_slice, tutu_lc10a_syn_slice_pca], axis=1)


# Convert coords
fig = plot_pca_transformed(curr_tutu_lc10a_syn_slice, tutu_lc10a_syn_slice_pca, lc10a_cdict,
                           xvar=f'{xvar}_post', yvar=f'{yvar}_post', hue_var='bodyId_post',
                           marker='o', markersize=50)
fig.suptitle(f'{zvar} slice: {z_start} to {z_end}')
                        
#Add shared colorbar
cbar_ax = fig.add_axes([0.91, 0.3, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('LO position (sorted order)')

putil.label_figure(fig, figid)
figname = f'{side}_pca_scores_slice'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)
#%%
# Bin PCA scores and plot
tutu_lc10a_syn_slice_pca_binned = bin_pca_scores(tutu_lc10a_syn_slice_pca)
#%
#%
bin_cmap = 'viridis_r'
fig = plot_joint_pca_scores(tutu_lc10a_syn_slice_pca_binned, tutu_lc10a_syn_slice_pca, 
                            lc10a_cdict, bin_cmap=bin_cmap,
                            markersize=20, marker='o', alpha=0.5,
                            hue_var='bodyId_post', 
                            marginal_marker='o', marginal_markersize=20)
putil.label_figure(fig, figid)
figname = f'{side}_pca_scores_binned'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)
