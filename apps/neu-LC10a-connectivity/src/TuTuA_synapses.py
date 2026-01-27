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
from neuprint import SynapseCriteria as SC

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
# Simple function to set colors - call this with different palette names to change colors
def set_segment_colors(segments, body_ids, palette='Viridis', n_colors=256):
    """Update color columns in segments DataFrame based on palette."""
    colors = []
    if isinstance(palette, str):
        palette_attr = getattr(bokeh.palettes, palette, None)
        if palette_attr is not None:
            if callable(palette_attr):
                colors = palette_attr(n_colors)
            elif isinstance(palette_attr, dict):
                colors = palette_attr.get(n_colors) or next(iter(palette_attr.values()))
            else:
                colors = list(palette_attr)
        if not colors:
            colors = sns.color_palette(palette, n_colors=n_colors).as_hex()
    elif isinstance(palette, (list, tuple)):
        colors = list(palette)
    else:
        colors = sns.color_palette(palette, n_colors=n_colors).as_hex()

    if len(colors) < len(body_ids):
        colors = sns.color_palette(colors, n_colors=len(body_ids)).as_hex()
    
    color_map = {bodyId: colors[int((i / len(body_ids)) * (len(colors) - 1))] 
                for i, bodyId in enumerate(body_ids)}
    
    # bodyId doesn't get a suffix because it's used as a merge key
    # Both child and parent have the same bodyId in each row
    segments['color_child'] = segments['bodyId'].map(color_map)
    segments['color_parent'] = segments['bodyId'].map(color_map)
    
    return segments

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


def plot_pca_transformed(tutu_lc10a_syn, tutu_lc10a_syn_pca, 
                        lc10a_cdict=None, hue_palette='viridis_r',
                        xvar='post_z', yvar='post_y', hue_var='post_root_id',
                        markersize=20, marker='x', invert_yaxis=True):
    if lc10a_cdict is None:
        colors = sns.color_palette(hue_palette, n_colors=len(sorted_lc10a_ids_list)).as_hex()
        lc10a_cdict = dict(zip(sorted_lc10a_ids_list, colors))

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
    # Add colorbar using lc10a color mapping so it reflects the palette order
    palette_colors = [lc10a_cdict.get(bodyId, '#000000') for bodyId in sorted_lc10a_ids_list]
    palette_cmap = ListedColormap(palette_colors)
    sm = cm.ScalarMappable(cmap=palette_cmap)
    sm.set_clim(0, max(len(palette_colors) - 1, 1))
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label(f'{hue_var}', fontsize=8)
    cbar.ax.tick_params(labelsize=8)

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

def plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, 
                        lc10a_cdict=None, hue_palette='viridis',
                        bin_cmap='viridis',
                        hue_var='post_root_id',
                        markersize=20, marker='x', alpha=1,
                        marginal_marker='o', marginal_markersize=20,
                        figsize=(5, 3.5)):
    #bin_cmap = 'viridis_r'
    
    max_bins = max(tutu_lc10a_syn_pca_binned['PC1_bin_label'].nunique(),
                   tutu_lc10a_syn_pca_binned['PC2_bin_label'].nunique(), 3)
    bin_palette = _palette_list_from_cmap(bin_cmap, n_colors=max_bins)

    # Plot the data in the new basis with PC1 and PC2
    fig, axn = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # Reduce white space by adjusting subplot layout more aggressively
    plt.subplots_adjust(top=0.98, bottom=0.12, hspace=0.3)

    # PC1 distribution (top subplot)
    ax_pc1 = axn[0]
    sns.scatterplot(data=tutu_lc10a_syn_pca_binned, 
                x='PC1_bin_numeric', y='PC1',
                hue='PC1_bin_label',
                palette=bin_palette, legend=0, ax=ax_pc1,
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
                palette=bin_palette, legend=0, ax=ax_pc2,
                marker=marginal_marker, s=marginal_markersize)
    ax_pc2.set_yticklabels([])
    ax_pc2.set_ylabel('')
    #ax_pc2.set_yticklabels([])
    ax_pc1.set_xticklabels([])

    # Only plot the max value for ax_pc1 and ax_pc2
    ax_pc1.set_ylim(0, tutu_lc10a_syn_pca_binned['PC1'].max())
    ax_pc2.set_xlim(0, tutu_lc10a_syn_pca_binned['PC2'].max())
    # Only label the max value for ax_pc1 and ax_pc2
    ax_pc1.set_yticks([tutu_lc10a_syn_pca_binned['PC1'].max()])
    ax_pc2.set_xticks([tutu_lc10a_syn_pca_binned['PC2'].max()])

    #plt.subplots_adjust(top=0.8)
    for ax in [ax_pc1, ax_main, ax_pc2]:
        sns.despine(ax=ax, top=True, right=True)

    return fig

def _palette_list_from_cmap(palette_cmap, n_colors=256):
    if isinstance(palette_cmap, str):
        return sns.color_palette(palette_cmap, n_colors=n_colors)
    if isinstance(palette_cmap, ListedColormap):
        return palette_cmap.colors or sns.color_palette("viridis", n_colors=n_colors)
    if hasattr(palette_cmap, "colors"):
        return list(palette_cmap.colors)
    if callable(palette_cmap):
        return palette_cmap(np.linspace(0, 1, n_colors))
    if isinstance(palette_cmap, (list, tuple)):
        return list(palette_cmap)
    return sns.color_palette("viridis", n_colors=n_colors)



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

#%%
# Get all LC10a neurons
LC10a_neurons, LC10a_roi_counts = neu.fetch_neurons(NC(type='LC10a',                                                          client=c))
LC10a_neurons.head()

#%%
# Get AOTU019 and AOTU025 neurons for reference
aotu19_neurons, aotu19_roi_counts = neu.fetch_neurons(NC(type='AOTU019', 
                                                         client=c))
aotu25_neurons, aotu25_roi_counts = neu.fetch_neurons(NC(type='AOTU025', 
                                                         client=c))
#%%
# Select 1 side to look at
side = 'L' #'R'

if side is not None:
    aotu19_ids = aotu19_neurons[aotu19_neurons['instance']==f'AOTU019_{side}']['bodyId'].unique()
    aotu25_ids = aotu25_neurons[aotu25_neurons['instance']==f'AOTU025_{side}']['bodyId'].unique()
    print(f"Number of AOTU019-{side} neurons: {len(aotu19_ids)}")
    print(f"Number of AOTU025-{side} neurons: {len(aotu25_ids)}")

    lc10a_ids = LC10a_neurons[LC10a_neurons['instance']==f'LC10a_{side}']['bodyId'].unique()
    print(f"Number of LC10a-{side} neurons: {len(lc10a_ids)}")
else:
    aotu19_ids = aotu19_neurons['bodyId'].unique()
    aotu25_ids = aotu25_neurons['bodyId'].unique()
    lc10a_ids = LC10a_neurons['bodyId'].unique()
    print(f"Number of AOTU019 neurons: {len(aotu19_ids)}")
    print(f"Number of AOTU025 neurons: {len(aotu25_ids)}")
    print(f"Number of LC10a neurons: {len(lc10a_ids)}")
#%%
# Get LC10a synapse
min_confidence = 0.95
syn_crit = SC(confidence=min_confidence)
lc10a_syn = neu.fetch_synapses(lc10a_ids, client=c,
                               nt='max', synapse_criteria=syn_crit)
#%
# Extract L or R from ROI(L) or ROI(R) using regexp to find what is inside the parentheses
lc10a_syn['side'] = npf.extract_side_from_column(lc10a_syn, column='roi')

print(f"Number of LC10a synapses: {len(lc10a_syn)}")
lc10a_syn.head()

#%%
# Get LC10a -> AOTU19/25 synapses
lc10a_aotu19_syn = neu.fetch_synapse_connections(lc10a_ids, aotu19_ids, client=c,
                                                  synapse_criteria=syn_crit)
lc10a_aotu25_syn = neu.fetch_synapse_connections(lc10a_ids, aotu25_ids, client=c,
                                                  synapse_criteria=syn_crit)
print(f"Number of LC10a -> AOTU19 synapses: {len(lc10a_aotu19_syn)}")
print(f"Number of LC10a -> AOTU25 synapses: {len(lc10a_aotu25_syn)}")

#%%
# PLOT 
# ======================================================

#%%
# Get LC10a LO coordinates and sort by y?
# -------------------------------------------------------------
if side is not None:
    LO_coords = lc10a_syn[(lc10a_syn['bodyId'].isin(lc10a_ids)
                    & (lc10a_syn['roi']==f'LO({side})'))].copy()
else:
    LO_coords = lc10a_syn[lc10a_syn['bodyId'].isin(lc10a_ids)
                        & (lc10a_syn['roi'].str.contains('LO'))].copy()

sort_by = 'z'
# Sort bodyids by sory_by
sorted_lc10a_ids = LO_coords.sort_values(by=sort_by, 
                        ascending=False)['bodyId'].unique()

#%%
# Create dictionary of colors
hue_palette = 'viridis'
sorted_lc10a_ids_list = list(sorted_lc10a_ids)
lc10a_colors = sns.color_palette(hue_palette, n_colors=len(sorted_lc10a_ids_list))
lc10a_cdict = dict(zip(sorted_lc10a_ids_list, lc10a_colors)) 

# Create a continuous mappable based on the order in sorted_lc10a_ids
lc10a_listed_cmap = ListedColormap(lc10a_colors)

aotu19_color = 'r'
aotu25_color = 'b'

#%%

# Color-code by LO position 
xvar = 'z'
yvar = 'y'
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=True, sharey=True)
ax=axn[0]
ax.set_title(f'LC10a-{side} synapses')
# also plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=f'{xvar}', y=f'{yvar}', #color='lightgray',
                s=10, hue='bodyId', palette=lc10a_cdict,
                legend=0)
ax=axn[1]
# also plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=f'{xvar}', y=f'{yvar}', color='lightgray',
                legend=0, s=5, alpha=0.5)
# plot aotu019 and 025
sns.scatterplot(data=lc10a_aotu19_syn, ax=ax,
                x=f'{xvar}_post', y=f'{yvar}_post', s=10, alpha=0.1,
                color=aotu19_color, label='AOTU19')
sns.scatterplot(data=lc10a_aotu25_syn, ax=ax,
                x=f'{xvar}_post', y=f'{yvar}_post', s=10, alpha=0.1,
                color=aotu25_color, label='AOTU25')
ax.legend(frameon=False, markerscale=5)
ax.set_aspect('equal')

ax.invert_yaxis()
#if xvar == 'x':
#    ax.invert_xaxis()

# save
putil.label_figure(fig, figid)
figname = f'reference_{side}_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)


#%%
# Get L10a neurons skeletons
# ======================================================    
# Create skeletons WITHOUT colors (so we can change colormapping later)
skeletons = []
for i, bodyId in enumerate(sorted_lc10a_ids):
    s = neu.fetch_skeleton(bodyId, format='pandas')
    s['bodyId'] = bodyId
    skeletons.append(s) 

skeletons = pd.concat(skeletons, ignore_index=True)
skeletons.head()

#%
# Join parent/child nodes for plotting as line segments below.
# (Using each row's 'link' (parent) ID, find the row with matching rowId.)
segments = skeletons.merge(skeletons, 'inner',
                           left_on=['bodyId', 'link'],
                           right_on=['bodyId', 'rowId'],
                           suffixes=['_child', '_parent'])


#%%
# BOKEH: Plot LC10a skeleton and synapses
# ======================================================
xvar = 'z'
yvar = 'y'
plot_aotu_syn = False
aotu_str = '_on_AOTU19-25' if plot_aotu_syn else ''

# Apply initial colors
set_segment_colors(segments, sorted_lc10a_ids,
                    palette=hue_palette)

# Plot skeletons
p = figure()
p.y_range.flipped = True
if xvar == 'x':
    p.x_range.flipped = True
if yvar == 'y':
    p.y_range.flipped = True
# Plot skeleton segments (in 2D)
seg_renderer = p.segment(x0=f'{xvar}_child', x1=f'{xvar}_parent',
                         y0=f'{yvar}_child', y1=f'{yvar}_parent',
                         color='color_child',
                         source=segments)

# To change colors later, just run these two lines:
# set_segment_colors(segments, sorted_lc10a_ids, palette='Cividis')  # Try: 'Viridis', 'Cividis', 'Plasma', 'Inferno'
# seg_renderer.data_source.data = {col: segments[col].values for col in segments.columns}
# label axes
p.xaxis.axis_label = xvar
p.yaxis.axis_label = yvar

if plot_aotu_syn:
    # Also plot the synapses from the above example
    # p.scatter(points['x_post'], points['z_post'], color=points['color'])
    p.scatter(lc10a_aotu19_syn[f'{xvar}_post'], 
            lc10a_aotu19_syn[f'{yvar}_post'], 
            color='red', size=2, alpha=0.7)
    p.scatter(lc10a_aotu25_syn[f'{xvar}_post'], 
            lc10a_aotu25_syn[f'{yvar}_post'], 
            color='blue', size=2, alpha=0.7) 

# title axis
p.title.text = f'LC10a-{side}, sorted by {sort_by}'

# set aspect ratio
p.aspect_ratio = 1

# Save as HTML only (avoid document ownership/show/export issues)
figname = f'skel_{side}_LC10a-synapses{aotu_str}_{xvar}-{yvar}'
#p.save(filename=os.path.join(output_dir, f'{figname}.png'))
#print(figname)

show(p)


# %%
# Get all TuTuA_2 neurons
# ======================================================
#src_type = 'AOTU042'
#TuTuA2_neurons, TuTuA2_roi_counts = neu.fetch_neurons(NC(type=src_type,                                                          client=c))
#TuTuA2_neurons.head()

# Get synapse connections from TuTuA_2 to LC10a neurons
# ======================================================
min_confidence = 0.95
syn_crit = SC(confidence=min_confidence)
min_total_weight = 10

#src = NC(type='TuTuA_2'
#src_type = 'TuTuA_2'
src_type = 'AOTU042'
#src_type = 'TuTuA_2'
# -------------------
src = NC(type=[src_type]) #'TuTuA_2', 'AOTU042']) #roi=f'LO({side})')
dst = NC(type='LC10a')

# bodyId_pre are the TuTuA_2 neurons
# bodyId_post are the LC10a neurons
tutu_lc10a_syn = neu.fetch_synapse_connections(src, dst, client=c,
                    nt='max',
                    min_total_weight=min_total_weight,
                    synapse_criteria=syn_crit)
# %
# Get TuTu-LC10a synapses on current side
tutu_lc10a_syn_side = tutu_lc10a_syn[tutu_lc10a_syn['bodyId_post'].isin(lc10a_ids)].copy().reset_index(drop=True)
print(f"Number of synapses to LC10a-{side} neurons: {tutu_lc10a_syn_side['bodyId_post'].nunique()}")

# add order from sorted_lc10a_ids
tutu_lc10a_syn_side[f'{sort_by}_order'] = tutu_lc10a_syn_side['bodyId_post'].map(dict(zip(sorted_lc10a_ids, range(len(sorted_lc10a_ids)))))
tutu_lc10a_syn_side.head()

# Add synapse counts per LC10a neuron
tutu_lc10a_syn_side['syn_count'] = tutu_lc10a_syn_side.groupby(['bodyId_post'])['bodyId_post'].transform('count')

lc10a_syn['syn_count'] = lc10a_syn.groupby(['bodyId'])['bodyId'].transform('count')

#%%
# Check source NT:
# -------------------------------------------------------------
xvar = 'y'
yvar = 'z'
pre_post = 'pre'
xvar_pre_post = f'{xvar}_{pre_post}'
yvar_pre_post = f'{yvar}_{pre_post}'

# Color by nt
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# Change size of points to match confidence_pre
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post,
                hue='nt', palette='colorblind', legend=1,
                #size=f'confidence_{pre_post}', 
                #sizes = (0.95, 1),
                edgecolor='none', 
                alpha=0.5)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1),
                frameon=False)
ax.set_title(f'{src_type}->LC10a ({side}), {pre_post}-synaptic',
             loc='left', fontsize=12)


#%%
# 1. Plot TuTuA_2 synapses on LC10a (And AOTU019/25 as reference)
# -------------------------------------------------------------
xvar = 'z'
yvar = 'y'
pre_post = 'pre'
hue_palette = 'viridis'
weight_palette = 'magma'

markersize= 50
alpha = 0.5
edgecolor = 'w'
lw=0.1

xvar_pre_post = f'{xvar}_{pre_post}'
yvar_pre_post = f'{yvar}_{pre_post}'
huevar = f'{sort_by}_order' #= f'{z}_{pre_post}'

fig, axn = plt.subplots(1, 2, figsize=(10, 5), 
                    sharex=True, sharey=True)
ax=axn[0]
# Plot L1a0 neurons
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color='lightgray', s=markersize, alpha=alpha)
# Plot TuTuA synapses
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post, 
                hue=huevar,
                palette=hue_palette, legend=0, 
                s=markersize, alpha=alpha)
title = f'{src_type}_{pre_post}, ({side}), hue=sorted LC10a {sort_by}-pos in LO'
ax.set_title(title, loc='left', fontsize=12) 

# Plot AOTU019 and 25 
ax=axn[1]
# LC10a terminals
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color='lightgray', s=5, alpha=0.5)
# AOTU019 and 25 terminals
sns.scatterplot(data=lc10a_aotu19_syn, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post, 
                color='red', s=markersize/2, alpha=alpha, label='AOTU019')
sns.scatterplot(data=lc10a_aotu25_syn, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post, 
                color='blue', s=markersize/2, alpha=alpha, label='AOTU025')
ax.legend(frameon=False)
ax.set_title('AOTU019 and 25', loc='left', fontsize=12)
ax.invert_yaxis()

#  zoom into AOTU
aotu_lims = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]    
x_min, x_max = aotu_lims[xvar].min(), aotu_lims[xvar].max()
y_min, y_max = aotu_lims[yvar].min(), aotu_lims[yvar].max()
print(f'AOTU limits: {x_min}, {x_max}, {y_min}, {y_max}')

for ax in axn:
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    if xvar == 'x':
        ax.invert_xaxis()

# save
putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}-synaptic_huesort-{sort_by}_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

# %
# 2. Color-code by LC10a synapse count
# -------------------------------------------------------------
fig, axn = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax=axn[0]
# LC10a terminals
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color='lightgray', s=5, alpha=0.5)
# Plot TuTuA synapses
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post, 
                hue=huevar, palette=hue_palette, legend=0, 
                s=markersize, alpha=alpha, 
                edgecolor=edgecolor, lw=lw)
# Add colorbar on this axis for hue variable
cbar_ax = fig.add_axes([0.45, 0.2, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=hue_palette)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('LO position')

fig.suptitle(f'{src_type}->LC10a ({side}), {pre_post}-synaptic')
ax.set_title(f'hue=sorted LC10a {sort_by}-pos in LO', loc='left', fontsize=12)

# 2. Plot by synapse
ax=axn[1]
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color='lightgray', s=5, alpha=0.5)
sns.scatterplot(data=tutu_lc10a_syn_side, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post, 
                hue='syn_count', palette=weight_palette, 
                legend=0, s=markersize, alpha=alpha,
                edgecolor=edgecolor, lw=lw)
ax.set_title(f'hue=syn_count by LC10a neuron', loc='left', fontsize=12)

# Add colorbar for syn_count
cbar_ax2 = fig.add_axes([0.91, 0.2, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=weight_palette)
sm.set_clim(tutu_lc10a_syn_side['syn_count'].min(), tutu_lc10a_syn_side['syn_count'].max())
cbar2 = fig.colorbar(sm, cax=cbar_ax2)
cbar2.set_label('Synapse count')

for ax in axn:
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.set_ylim([12000, 20000])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    if xvar == 'x':
        ax.invert_xaxis()
plt.subplots_adjust(wspace=0.5)

# save
putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}-synaptic_syn_count_huesort-{sort_by}_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# Plot joint distribution of TuTu-LC10a synapses
# -------------------------------------------------------------
g = sns.jointplot(data=tutu_lc10a_syn_side, 
                  x=xvar_pre_post, y=yvar_pre_post, 
                 kind='hist') #, bins=20)
g.ax_joint.set_aspect('equal')
g.ax_joint.invert_yaxis()


#%%
# PCA on synapses (TuTu->LC10a)
# =======================================================
# Filter to only synapses to sorted LC10a neurons
#tutu_lc10a_syn_side = tutu_lc10a_syn[tutu_lc10a_syn['bodyId_post'].isin(sorted_lc10a_ids)].copy().reset_index(drop=True)

#%%
xvar = 'z'
yvar = 'y'
pre_post = 'pre'
xvar_pre_post = f'{xvar}_{pre_post}'
yvar_pre_post = f'{yvar}_{pre_post}'
# Convert coords
lc10a_pca_scores = do_pca_on_synapses(tutu_lc10a_syn_side, 
                        xvar=xvar_pre_post, 
                        yvar=yvar_pre_post)

#% # Add PCA scores to the original dataframe
tutu_lc10a_syn_pca = pd.concat([tutu_lc10a_syn_side, lc10a_pca_scores], axis=1)

fig = plot_pca_transformed(tutu_lc10a_syn_side, 
                            tutu_lc10a_syn_pca, 
                           #hue_palette=hue_palette,
                           lc10a_cdict,
                           xvar=xvar_pre_post, 
                           yvar=yvar_pre_post, 
                           hue_var='bodyId_post',
                           marker='o', markersize=markersize)
 
putil.label_figure(fig, figid)
figname = f'check_pca_{src_type}-{side}_{pre_post}_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# Bin PCA scores and plot
n_pca_bins = 10
tutu_lc10a_syn_pca_binned = bin_pca_scores(tutu_lc10a_syn_pca,
                                n_bins=n_pca_bins)
# Plot joint and marginal distributions
fig = plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, 
                            lc10a_cdict, 
                            bin_cmap='viridis_r',
                            markersize=10, marker='o',
                            hue_var='bodyId_post', 
                            marginal_marker='o', marginal_markersize=20,
                            figsize=(4.5,4))
putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}_pca_scores_binned'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# QC: Check each bin
# Plot the synapses for each PC1 bin, x-y view
pc_bins = tutu_lc10a_syn_pca['PC1_bin_label'].unique()

lc10a_syn_zoom = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]
nr = 3
nc = int(np.ceil((len(pc_bins)-1) / nr))
fig, axn = plt.subplots(nr, nc, #1, 
                        figsize=(nc*1.2, nr*1.2),
                       sharex=True, sharey=True)
for i, (pbin, pdf) in enumerate(tutu_lc10a_syn_pca.groupby('PC1_bin_label')):
    ax=axn.flat[i]
    # plot lc10a first
    sns.scatterplot(data=lc10a_syn_zoom, ax=ax,
                    x=xvar, y=yvar,
                    color='lightgray',
                    s=1, edgecolor='none', alpha=0.5)
    sns.scatterplot(data=pdf, ax=ax,
                    x=xvar_pre_post, y=yvar_pre_post,
                    hue='bodyId_post',
                    palette=lc10a_cdict, legend=0,
                    s=5, edgecolor='none')
    ax.set_aspect('equal')
    ax.set_title(f'{pbin:0.1f}', fontsize=8, loc='left')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if i==0:
        ax.set_ylabel(f'{yvar}')
        ax.set_xlabel(f'{xvar}')
    else: #en(pc_bins)-2:
        ax.set_ylabel('')
        ax.set_xlabel('')
ax.invert_yaxis()
ax.invert_xaxis()

# colorbar
cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('LO position')

# Remove empty axes
for ax in axn.flat[i+1:]:
    ax.remove()
plt.subplots_adjust(hspace=0.8)


fig.suptitle(f'{src_type}->LC10a synapses, PC1 bins',
             fontsize=8)

putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}_PC1_bins_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)


#%%
min_confidence = 0.95
tutu_syn_confident = tutu_lc10a_syn_pca[(tutu_lc10a_syn_pca['confidence_pre']>=min_confidence)
                            & (tutu_lc10a_syn_pca['confidence_post']>=min_confidence)]
tutu_syn_confident['PC1_bin_numeric'] = pd.to_numeric(tutu_syn_confident['PC1_bin_label'])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.countplot(data=tutu_syn_confident, ax=ax,
                x='PC1_bin_label')
ax.set_xlabel('PC1')

#%%
fig, axn = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(f'{src_type}({side})-{pre_post}, x={xvar}, y={yvar}')
ax=axn[0, 0]
# Plot PC1 vs. PC2, color by LO position
sns.scatterplot(data=tutu_syn_confident, ax=ax,
                x='PC1', y='PC2',
                hue='bodyId_post', palette=lc10a_cdict,
                alpha=0.5, legend=0, s=50)

ax=axn[0, 1]
# Plot syn_count (per LC10a neuron) vs. position along PC1
sns.scatterplot(data=tutu_syn_confident, ax=ax,
                x='PC1', y='syn_count',
                hue='PC1_bin_label', #bodyId_post',
                alpha=0.5,
                legend=0)
sns.regplot(data=tutu_syn_confident, ax=ax,
                x='PC1', y='syn_count',
                scatter=False)
ax.set_xlabel('PC1')

ax=axn[1, 0]
# Plot counts of synapses along PC1
sns.countplot(data=tutu_syn_confident, ax=ax,
                x='PC1_bin_numeric')

ax=axn[1, 1]
# Plot syn_count (per PC1 bin) vs. PC1 bin label
sns.pointplot(data=tutu_syn_confident, ax=ax,
                x='PC1_bin_numeric', y='syn_count',
                alpha=0.5,
                legend=0)
# Clean up xtick labels
for ax in axn[1, :]:
    ax.set_xticklabels([])


fig.suptitle(f'TuTuA2({side})-{pre_post}, x={xvar}, y={yvar}',
             fontsize=12)
putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}_PC1_bins_counts_vs_syncounts_{xvar}-{yvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)


#%%

# Bin y-order
n_bins = 10
bins = np.linspace(tutu_syn_confident[f'{sort_by}_order'].min(), tutu_syn_confident[f'{sort_by}_order'].max(), n_bins)
tutu_syn_confident[f'{sort_by}_order_bin'] = pd.cut(tutu_syn_confident[f'{sort_by}_order'], bins)
tutu_syn_confident[f'{sort_by}_order_bin_label'] = tutu_syn_confident[f'{sort_by}_order_bin'].apply(lambda x: x.mid)

# Plot syn_count (per y-order bin) vs. y-order bin label
fig, axn = plt.subplots(1, 2, figsize=(6, 4))
ax=axn[0]
lc10a_syn_zoom = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]
sns.scatterplot(data=lc10a_syn_zoom, ax=ax,
                x=xvar, y=yvar,
                color='lightgray', s=5, alpha=0.5)
sns.scatterplot(data=tutu_syn_confident, ax=ax,
                x=xvar_pre_post, y=yvar_pre_post,
                hue='bodyId_post', palette=lc10a_cdict, legend=0,
                s=5, edgecolor='none')
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
putil.remove_spines(ax, axes=['right', 'top', 'bottom', 'left'])

ax=axn[1]
sns.countplot(data=tutu_syn_confident, ax=ax,
                x=f'{sort_by}_order_bin_label',
                hue=f'{sort_by}_order_bin_label',
                palette='viridis_r', legend=0,
                alpha=0.5)
# Annotate by how many values per bin
for i, (v, tmp) in enumerate(tutu_syn_confident.groupby(f'{sort_by}_order_bin_label')):
    ax.text(i, 10, f'{tmp["bodyId_post"].nunique()}',
            ha='center', va='bottom', fontsize=8)
# Format x ticks
ax.set_xticklabels([f'{int(round(float(x)))}' for x in ax.get_xticks()])
ax.set_box_aspect(1)
sns.despine(ax=ax, offset=4, trim=True)

plt.subplots_adjust(wspace=0.5)

# save
putil.label_figure(fig, figid)

#%%






#%%
# Reslice 2D to match 2p imaging?

def slice_coordinates(curr_lc10a_syn, zvar='y', z_steps=5):

    zmin, zmax = curr_lc10a_syn[f'{zvar}'].min(), curr_lc10a_syn[f'{zvar}'].max()
    #print( f'z range: {zmin} to {zmax}')
    z_depth = zmax - zmin
    slice_list = []
    for i in range(z_steps):
        curr_z_slice = [zmin + i*z_depth/z_steps, zmin + (i+1)*z_depth/z_steps]
        #print(curr_z_slice)
        #curr_lc10a_syn.loc[curr_lc10a_syn[f'{zvar}'].between(curr_z_slice[0], curr_z_slice[1]), f'{yvar}_slice'] = i
        #curr_lc10a_syn.loc[curr_lc10a_syn[f'{zvar}'].between(curr_z_slice[0], curr_z_slice[1]), f'{yvar}_range'] = (curr_z_slice[0], curr_z_slice[1])
        curr_slice = curr_lc10a_syn[curr_lc10a_syn[f'{zvar}'].between(curr_z_slice[0], curr_z_slice[1])].copy()
        curr_slice['z_slice'] = i
        curr_slice['z_range_start'] = curr_z_slice[0]
        curr_slice['z_range_end'] = curr_z_slice[1]
        slice_list.append(curr_slice)     

    curr_lc10a_syn = pd.concat(slice_list)

    return curr_lc10a_syn

# %%
#tutu_lc10a_syn_slice

# FIND THE CORRECT SLICING TO MATCH 2p - LC10a terminals
# -------------------------------------
xvar = 'x'
yvar = 'z'
zvar = 'y'
curr_side = side

lc10a_syn_conf = lc10a_syn[(lc10a_syn['roi']==f'AOTU({curr_side})')
                            & (lc10a_syn['confidence']>=min_confidence)]
tutu_lc10a_syn_conf = tutu_lc10a_syn_side[(tutu_lc10a_syn_side['roi_pre'].str.contains(curr_side))
                            & (tutu_lc10a_syn_side['confidence_pre']>=min_confidence)]

z_steps = 5
zvar = 'y'

lc10a_syn_conf = slice_coordinates(lc10a_syn_conf, 
                                    zvar=zvar, z_steps=z_steps)
tutu_lc10a_syn_conf = slice_coordinates(tutu_lc10a_syn_conf, 
                                    zvar=f'{zvar}_pre', z_steps=z_steps)

# Split AOTU into z slices:
#%%
markersize = 10
alpha = 1

fig, axn = plt.subplots(1, z_steps, figsize=(4*z_steps, 6),
                        sharex=True, sharey=True)
ri = 0
lc10a_syn_xlim = lc10a_syn_conf[f'{xvar}'].min(), lc10a_syn_conf[f'{xvar}'].max()
lc10a_syn_ylim = lc10a_syn_conf[f'{yvar}'].min(), lc10a_syn_conf[f'{yvar}'].max()

for i, (v, lc_slice_) in enumerate(lc10a_syn_conf.groupby('z_slice')):
    #curr_z_slice = [zmin + i*z_depth/z_steps, zmin + (i+1)*z_depth/z_steps]
    ax=axn[i]
    # add lc10a 
    sns.scatterplot(data=lc10a_syn_conf, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    color='lightgray',
                    s=markersize, edgecolor='none', alpha=alpha)
    # overlay color-coded by LO position
    sns.scatterplot(data=lc_slice_, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    hue='bodyId', palette=lc10a_cdict, legend=0,
                    s=markersize, edgecolor='none', alpha=alpha)
    curr_z_slice = (lc_slice_['z_range_start'].iloc[0], lc_slice_['z_range_end'].iloc[0])
    ax.set_title(f'{zvar} slice: {curr_z_slice[0]} to {curr_z_slice[1]}')
    # zoom in
    ax.set_xlim(lc10a_syn_xlim)
    ax.set_ylim(lc10a_syn_ylim)

for ax in axn:
    ax.set_aspect('equal')
ax.invert_yaxis()

# Sared colorbar
cbar_ax = fig.add_axes([0.91, 0.3, 0.01, 0.3])
sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('LO position (sorted order)')

#%%

lc_markersize = 10
tutu_markersize = 50
alpha = 0.5

lc10a_syn_zoom = lc10a_syn_conf[lc10a_syn_conf['roi'].str.contains('AOTU')]
lc10a_syn_zoom_xlim = lc10a_syn_zoom[f'{xvar}'].min(), lc10a_syn_zoom[f'{xvar}'].max()
lc10a_syn_zoom_ylim = lc10a_syn_zoom[f'{yvar}'].min(), lc10a_syn_zoom[f'{yvar}'].max()


fig, axn = plt.subplots(2, z_steps, figsize=(4*z_steps, 6),
                        sharex=True, sharey=True)

for i, (v, lc_slice_) in enumerate(lc10a_syn_conf.groupby('z_slice')):
    curr_z_slice = (lc_slice_['z_range_start'].iloc[0], lc_slice_['z_range_end'].iloc[0])
    ax=axn[0, i]
    # all lc10a 
    if i==0:
        curr_title = f'LC10a synapses\n{zvar} slice: {curr_z_slice[0]:0.1f} to {curr_z_slice[1]:0.1f}'
        ax.set_title(curr_title, loc='left')
    else:
        ax.set_title(f'{zvar} slice: {curr_z_slice[0]:0.1f} to {curr_z_slice[1]:0.1f}', loc='left')

    sns.scatterplot(data=lc10a_syn_zoom, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    color='lightgray',
                    s=lc_markersize, edgecolor='none', alpha=alpha)
    # color current slice by LO bodyID
    sns.scatterplot(data=lc_slice_, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    hue='bodyId', palette=lc10a_cdict, legend=0,
                    s=lc_markersize, edgecolor='none', alpha=alpha)
    
    if i == z_steps-1:
        # Add colorbar shared only for TOP row of subplots:
        cbar_ax = fig.add_axes([0.91, 0.6, 0.01, 0.25])
        sm = cm.ScalarMappable(cmap=lc10a_listed_cmap)
        sm.set_clim(0, len(sorted_lc10a_ids_list) - 1)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('LO position')


    ax=axn[1, i]
    #tutu_lc10a_syn_slice = curr_tutu_lc10a_syn[curr_tutu_lc10a_syn[f'{zvar}_pre'].between(curr_z_slice[0], curr_z_slice[1])]
    #curr_tutu_lc10a_syn.loc[curr_tutu_lc10a_syn[f'{zvar}_pre'].between(curr_z_slice[0], curr_z_slice[1]), 'z_slice'] = i
    tutu_slice_ = tutu_lc10a_syn_conf[tutu_lc10a_syn_conf['z_slice']==i]
    # all lc10a 
    if i==0:
        curr_title = f'{src_type} synapses\n{zvar} slice: {curr_z_slice[0]:0.1f} to {curr_z_slice[1]:0.1f}'
        ax.set_title(curr_title, loc='left')
    else:
        ax.set_title(f'{zvar} slice: {curr_z_slice[0]:0.1f} to {curr_z_slice[1]:0.1f}', loc='left')

    sns.scatterplot(data=lc10a_syn_conf, ax=ax,
                    x=f'{xvar}', y=f'{yvar}',
                    color='lightgray',
                    s=lc_markersize, edgecolor='none', alpha=alpha)
    # current TuTuA synapses
    sns.scatterplot(data=tutu_slice_, ax=ax,
                    x=f'{xvar}_pre', y=f'{yvar}_pre',
                    hue='syn_count', 
                    palette=weight_palette, legend=0,
                    s=tutu_markersize, edgecolor='none', alpha=alpha)

    if i == z_steps-1:
        # Add colorbar shared only for BOTTOM row of subplots:
        cbar_ax = fig.add_axes([0.91, 0.2, 0.01, 0.25])
        sm = cm.ScalarMappable(cmap=weight_palette)
        sm.set_clim(vmin=0, #tutu_lc10a_syn_conf['syn_count'].min(), 
                    vmax=tutu_lc10a_syn_conf['syn_count'].max())
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Per Neuron Synapse Count')

# axes
for ax in axn.flat:
    ax.set_aspect('equal')
ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_{pre_post}_by_{xvar}-{yvar}-slice-{zvar}'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%

# Look at 1 slice
z_slice = 1
lc10a_slice = lc10a_syn_conf[lc10a_syn_conf['z_slice']==z_slice].reset_index(drop=True)
tutu_slice = tutu_lc10a_syn_conf[tutu_lc10a_syn_conf['z_slice']==z_slice].reset_index(drop=True)

# Do pca on this slice
curr_pca = do_pca_on_synapses(tutu_slice, xvar=f'{xvar}_pre', yvar=f'{yvar}_pre')   

# Add PC scores to the original dataframe
curr_pca = pd.concat([tutu_slice, curr_pca], axis=1)
# plot
fig = plot_pca_transformed(tutu_slice, curr_pca, 
                           lc10a_cdict,
                           xvar=f'{xvar}_pre', yvar=f'{yvar}_pre', 
                           hue_var='bodyId_post', 
                           marker='o', markersize=50)
fig.suptitle(f'{zvar} slice: {z_slice}')





#%%
# Total N synapses per z-slice
# -------------------------------------
fig, axn = plt.subplots(1, 3, figsize=(9, 3),
                        sharex=True, sharey=False) 
# all lc10a
ax=axn[0]
sns.countplot(data=lc10a_syn_zoom, ax=ax,
                x='z_slice',
                hue='z_slice',
                palette='viridis_r', legend=0,
                alpha=0.5)
ax.set_title('Total LC10a synapses by slice', loc='left',
             fontsize=10)
ax=axn[1]
sns.countplot(data=tutu_lc10a_syn_conf, ax=ax,
                x='z_slice',
                hue='z_slice',
                palette='viridis_r', legend=0,
                alpha=0.5)
ax.set_title('Total TuTu synapses by slice', loc='left',
             fontsize=10)

# Calculate ratio of TuTu synapses to LC10a synapses by slice
ratios = []
for z_, lc_z in lc10a_syn_zoom.groupby('z_slice'):
    tu_z = tutu_lc10a_syn_conf[tutu_lc10a_syn_conf['z_slice']==z_]
    #ratio = tu_z['syn_count'].sum() / lc_z['syn_count'].sum()
    tu_summed_per_neuron = tu_z.groupby('bodyId_post')['syn_count'].unique().sum()
    lc_summed_per_neuron = lc_z.groupby('bodyId')['syn_count'].unique().sum()
    ratio_total = tu_z['bodyId_post'].count() / lc_z['bodyId'].count()
    ratios_tmp = {'ratio_per_neuron': tu_summed_per_neuron / lc_summed_per_neuron, 
                'ratio_total': ratio_total,
                'z_slice': z_}
    ratios_ = pd.DataFrame(ratios_tmp)
    ratios.append(ratios_)
ratio_df = pd.concat(ratios, ignore_index=True)

# plot
ax = axn[2]
sns.pointplot(data=ratio_df, ax=ax,
                x='z_slice', y='ratio_total',
                alpha=0.5, 
                errorbar='se')

for ax in axn:
    ax.set_box_aspect(1)

plt.subplots_adjust(wspace=0.5)

putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_ratio_total'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)

#%%
# N synapses per neuron per z-slice
# -------------------------------------
lc10a_syn_counts = lc10a_syn_zoom.groupby('z_slice')['bodyId'].value_counts().reset_index()
tutu_syn_counts = tutu_lc10a_syn_conf.groupby('z_slice')['bodyId_post'].value_counts().reset_index()

fig, axn = plt.subplots(1, 3, figsize=(9, 3),
                        sharex=True, sharey=False)
ax=axn[0]
sns.pointplot(data=lc10a_syn_counts, ax=ax,
                 x='z_slice', y='count',
                 alpha=0.5)
ax.set_title('LC10a synapses per neuron by slice', fontsize=10)
ax=axn[1]
sns.pointplot(data=tutu_syn_counts, ax=ax,
                x='z_slice', y='count',
                alpha=0.5)
ax.set_title('TuTu synapses per neuron by slice', fontsize=10)

#%
# Calculate ratio
r_list = []
for z_, lc_z in lc10a_syn_counts.groupby('z_slice'):
#lc_z = lc10a_syn_counts[lc10a_syn_counts['z_slice']==z_]
    tu_z = tutu_syn_counts[tutu_syn_counts['z_slice']==z_]

    # get common IDs
    common_ids = set(lc_z['bodyId']).intersection(set(tu_z['bodyId_post']))
    lc_z_common = lc_z[lc_z['bodyId'].isin(common_ids)]
    tu_z_common = tu_z[tu_z['bodyId_post'].isin(common_ids)]
    lc_z_common.index = lc_z_common['bodyId']
    tu_z_common.index = tu_z_common['bodyId_post']
    # Caclulate ratio per neuron
    id_list = lc_z_common.index.tolist()
    ratio = tu_z_common.loc[id_list]['count'] / lc_z_common.loc[id_list]['count']
    ratio = ratio.reset_index().rename(columns={'count': 'ratio'})
    ratio['z_slice'] = z_

    r_list.append(ratio)
ratios = pd.concat(r_list, ignore_index=True)

#%
ax = axn[2]
sns.pointplot(data=ratios, ax=ax,
                x='z_slice', y='ratio',
                alpha=0.5, 
                errorbar='se')
ax.set_xlabel('z_slice')
ax.set_ylabel('ratio')
ax.set_title(f'{src_type}:LC10a per neuron', fontsize=10)

plt.subplots_adjust(wspace=0.5)
for ax in axn:
    ax.set_box_aspect(1)

# save
putil.label_figure(fig, figid)
figname = f'{src_type}-{side}_ratio_per_neuron'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)
#%%
import matplotlib.colors as mcolors
 # Check if the syanpses actually match the bin
n_bins = tutu_lc10a_syn_side['zvar_bin_label'].nunique()
fig, axn = plt.subplots(n_bins, 1, 
                    figsize=(5, 1*n_bins), 
                    sharex=True, sharey=True)

syn_count_range = tutu_lc10a_syn_side['syn_count'].min(), tutu_lc10a_syn_side['syn_count'].max()
vmin, vmax = syn_count_range
hue_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
for i, (v, tmp) in enumerate(tutu_lc10a_syn_side.groupby('zvar_bin_label')):
    ax = axn[i]
    sns.scatterplot(data=tmp, ax=ax,
                x=f'{xvar}_post', 
                y=f'{yvar}_post',
                hue='bodyId_post', 
                palette=lc10a_cdict, legend=0, 
                hue_norm=hue_norm,
                s=20, alpha=1)
    ax.set_aspect('equal')
    curr_zrange = tmp[f'{zvar}_pre'].min(), tmp[f'{zvar}_pre'].max()
    ax.set_title(f'{zvar} slice: {curr_zrange[0]} to {curr_zrange[1]}')
ax.invert_yaxis()


#%%

# %%

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
fig = plot_pca_transformed(curr_tutu_lc10a_syn_slice, 
                            tutu_lc10a_syn_slice_pca, 
                            lc10a_cdict,
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
fig = plot_joint_pca_scores(tutu_lc10a_syn_slice_pca_binned, tutu_lc10a_syn_slice_pca, 
                            lc10a_cdict, 
                            bin_cmap=hue_palette,
                            markersize=20, marker='o', alpha=0.5,
                            hue_var='bodyId_post', 
                            marginal_marker='o', marginal_markersize=20)
putil.label_figure(fig, figid)
figname = f'{side}_pca_scores_binned'
putil.save_fig(figname, fig, figid, output_dir)
print(figname)
