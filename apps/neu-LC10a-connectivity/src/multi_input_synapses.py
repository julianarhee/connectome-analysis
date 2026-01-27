#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Juliana Rhee
@ Filename: multi_input_synapses.py
@ Create Time: 2026-01-26 10:53:42
@ Modified by: Juliana Rhee
@ Modified time: 2026-01-26 10:53:48
@ Description: Get the synapses of the TuTuA neurons to LC10a neurons.

'''
#%%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D

# import neuprint stuff
import neuprint as neu
from neuprint import NeuronCriteria as NC
from neuprint import SynapseCriteria as SC

import bokeh.palettes
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import export_png, export_svgs, output_file, save
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

def bokeh_plot_settings(p, min_fontsize=12):
    p.background_fill_color = None
    p.border_fill_color = None
    p.outline_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_font_style = "normal"
    p.axis.major_label_text_color = "black"
    # enforce consistent font family/size for SVG export
    p.axis.major_label_text_font = 'Arial' #"DejaVu Sans"
    p.axis.major_label_text_font_size = f"{min_fontsize}pt"
    p.axis.axis_label_text_font = 'Arial' #"DejaVu Sans"
    p.axis.axis_label_text_font_size = f"{min_fontsize+1}pt"
    return p

#%% Output dir
rootdir = '/Volumes/Juliana/connectome'
output_dir = os.path.join(rootdir, 'analyses', 'neuprint', 
                            'multi_input_synapses')

# Make output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Output directory: {output_dir}')

#%%
# Get all LC10a neurons
LC10a_neurons, LC10a_roi_counts = neu.fetch_neurons(NC(type='LC10a',                                                          client=c))
LC10a_neurons.head()

#%%
side = 'L' #'R'

if side is not None:
    lc10a_ids = LC10a_neurons[LC10a_neurons['instance']==f'LC10a_{side}']['bodyId'].unique()
    print(f"Number of LC10a-{side} neurons: {len(lc10a_ids)}")
else:
    lc10a_ids = LC10a_neurons['bodyId'].unique()
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

lc10a_syn['syn_count'] = lc10a_syn.groupby(['bodyId'])['bodyId'].transform('count')

print(f"Number of LC10a synapses: {len(lc10a_syn)}")
lc10a_syn.head()

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
#
#%
# Create dictionary of colors
hue_palette = 'viridis'
sorted_lc10a_ids_list = list(sorted_lc10a_ids)
lc10a_colors = sns.color_palette(hue_palette, n_colors=len(sorted_lc10a_ids_list))
lc10a_cdict = dict(zip(sorted_lc10a_ids_list, lc10a_colors)) 

# Create a continuous mappable based on the order in sorted_lc10a_ids
lc10a_listed_cmap = ListedColormap(lc10a_colors)

# %%
def get_input_synapses(src_type, dst='LC10a',
                        sorted_lc10a_ids=None,
                       min_total_weight=10,
                       min_confidence=0.95):
    # -------------------
    syn_crit = SC(confidence=min_confidence)
    src = NC(type=[src_type]) 
    dst = NC(type=dst)

    # bodyId_pre are the TuTuA_2 neurons
    # bodyId_post are the LC10a neurons
    aotu42_lc10a_syn = neu.fetch_synapse_connections(src, dst, client=c,
                        nt='max',
                        min_total_weight=min_total_weight,
                        synapse_criteria=syn_crit)
    # %
    if sorted_lc10a_ids is not None:
        # Get TuTu-LC10a synapses on current side
        aotu42_lc10a_syn_side = aotu42_lc10a_syn[aotu42_lc10a_syn['bodyId_post'].isin(sorted_lc10a_ids)].copy().reset_index(drop=True)
        #print(f"Number of synapses to LC10a-{side} neurons: {aotu42_lc10a_syn_side['bodyId_post'].nunique()}")

    else:
        aotu42_lc10a_syn_side = aotu42_lc10a_syn.copy().reset_index(drop=True)

    # add order from sorted_lc10a_ids
    aotu42_lc10a_syn_side[f'{sort_by}_order'] = aotu42_lc10a_syn_side['bodyId_post'].map(dict(zip(sorted_lc10a_ids, range(len(sorted_lc10a_ids)))))
    #aotu42_lc10a_syn_side.head()

    # Add synapse counts per LC10a neuron
    aotu42_lc10a_syn_side['syn_count'] = aotu42_lc10a_syn_side.groupby(['bodyId_post'])['bodyId_post'].transform('count')

    return aotu42_lc10a_syn_side

def get_segments(bodyIds):
    skeletons = []
    for i, bodyId in enumerate(bodyIds):
        s = neu.fetch_skeleton(bodyId, format='pandas')
        s['bodyId'] = bodyId
        skeletons.append(s) 
    skeletons = pd.concat(skeletons, ignore_index=True)
    #%
    # Join parent/child nodes for plotting as line segments below.
    # (Using each row's 'link' (parent) ID, find the row with matching rowId.)
    segments = skeletons.merge(skeletons, 'inner',
                           left_on=['bodyId', 'link'],
                           right_on=['bodyId', 'rowId'],
                           suffixes=['_child', '_parent'])
    return segments

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


def plot_skeleton_matplotlib(segments, body_ids, xvar='z', yvar='y',
                             palette='viridis', figsize=(6, 6), ax=None,
                             linewidth=0.5, aspect='equal',
                             invert_y=True, label_axes=True):
    """Draw skeleton segments with Matplotlib instead of Bokeh."""
    segments = set_segment_colors(segments.copy(), body_ids, palette=palette)
    coords = segments[[f'{xvar}_child', f'{yvar}_child',
                       f'{xvar}_parent', f'{yvar}_parent',
                       'color_child']].dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    lines = coords[[f'{xvar}_child', f'{yvar}_child',
                    f'{xvar}_parent', f'{yvar}_parent']].to_numpy()
    if lines.size == 0:
        return fig, ax

    lines = lines.reshape(-1, 2, 2)
    colors = coords['color_child'].tolist()
    lc = LineCollection(lines, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    ax.autoscale()

    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect(aspect)
    if label_axes:
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)

    return fig, ax


def compute_synapse_overlaps(aotu_df, tutu_df, radius=500,
                             xcol='x_pre', ycol='y_pre', zcol='z_pre'):
    """Return synapse pairs whose distance is <= radius."""
    if aotu_df.empty or tutu_df.empty:
        return pd.DataFrame(columns=[
            'bodyId_post', 'index_aotu', 'index_tutu',
            'x_aotu', 'y_aotu', 'z_aotu',
            'x_tutu', 'y_tutu', 'z_tutu', 'distance'])

    a_points = aotu_df[[xcol, ycol, zcol]].to_numpy()
    t_points = tutu_df[[xcol, ycol, zcol]].to_numpy()
    if a_points.size == 0 or t_points.size == 0:
        return pd.DataFrame(columns=[
            'bodyId_post', 'index_aotu', 'index_tutu',
            'x_aotu', 'y_aotu', 'z_aotu',
            'x_tutu', 'y_tutu', 'z_tutu', 'distance'])

    tree = cKDTree(t_points)
    neighbors = tree.query_ball_point(a_points, r=radius)
    records = []
    for a_idx, t_idxs in enumerate(neighbors):
        for t_idx in t_idxs:
            dist = np.linalg.norm(a_points[a_idx] - t_points[t_idx])
            records.append({
                'bodyId_post': aotu_df.iloc[a_idx]['bodyId_post'],
                'index_aotu': a_idx,
                'index_tutu': t_idx,
                'x_aotu': a_points[a_idx, 0],
                'y_aotu': a_points[a_idx, 1],
                'z_aotu': a_points[a_idx, 2],
                'x_tutu': t_points[t_idx, 0],
                'y_tutu': t_points[t_idx, 1],
                'z_tutu': t_points[t_idx, 2],
                'distance': dist
            })
    return pd.DataFrame(records)

#%%
# Get synapse connections from TuTuA_2 to LC10a neurons
# ======================================================
min_confidence = 0.95
min_total_weight = 10

src_type = 'AOTU042'
aotu42_lc10a_syn = get_input_synapses(src_type, dst='LC10a',
                        sorted_lc10a_ids=sorted_lc10a_ids,
                       min_total_weight=10,
                       min_confidence=0.95)


# %%
tutuA2_lc10a_syn = get_input_synapses(src_type='TuTuA_2', dst='LC10a',
                        sorted_lc10a_ids=sorted_lc10a_ids,
                       min_total_weight=10,
                       min_confidence=0.95)

#%%
# For plotting
lc10a_syn_zoom = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]

lc10a_color = 'lightgray'
aotu42_color = 'red'
tutu_color = 'blue'

lc10a_segments = get_segments(lc10a_syn['bodyId'].unique())
lc10a_segments = set_segment_colors(lc10a_segments, lc10a_syn['bodyId'].unique(), palette=hue_palette)

lc10a_segments.head()

# %%
# Get L10a neurons skeletons
# ======================================================    
# Create skeletons WITHOUT colors (so we can change colormapping later)
markersize=2
curr_lc10a_ids = [#sorted_lc10a_ids[0],
                  sorted_lc10a_ids[1], 
                  #sorted_lc10a_ids[2],
                  #sorted_lc10a_ids[3],
                  #sorted_lc10a_ids[4],
                  sorted_lc10a_ids[45],
                  #sorted_lc10a_ids[46],
                  #sorted_lc10a_ids[47],
                  #sorted_lc10a_ids[48],
                  #sorted_lc10a_ids[49],
                  #sorted_lc10a_ids[50],
                  #sorted_lc10a_ids[51],
                  #sorted_lc10a_ids[52],
                  #sorted_lc10a_ids[53],
                  #sorted_lc10a_ids[-2],
                  #sorted_lc10a_ids[-1],
                  #sorted_lc10a_ids[-3],
                  #sorted_lc10a_ids[-4],
                  sorted_lc10a_ids[-5]]

example_segments = get_segments(curr_lc10a_ids)
# skeletons = []
# for i, bodyId in enumerate(curr_lc10a_ids):
#     s = neu.fetch_skeleton(bodyId, format='pandas')
#     s['bodyId'] = bodyId
#     skeletons.append(s) 
# 
# skeletons = pd.concat(skeletons, ignore_index=True)
# skeletons.head()
# #%
# # Join parent/child nodes for plotting as line segments below.
# # (Using each row's 'link' (parent) ID, find the row with matching rowId.)
# segments = skeletons.merge(skeletons, 'inner',
#                            left_on=['bodyId', 'link'],
#                            right_on=['bodyId', 'rowId'],
#                            suffixes=['_child', '_parent'])
 


#%%
#import bokeh.models.legend
from bokeh.models import Legend, LegendItem
zoom = False

# %
xvar = 'z'
yvar = 'y'
markersize = 5

# Apply initial colors
set_segment_colors(example_segments, sorted_lc10a_ids,
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
                         source=example_segments)

# To change colors later, just run these two lines:
# set_segment_colors(segments, sorted_lc10a_ids, palette='Cividis')  # Try: 'Viridis', 'Cividis', 'Plasma', 'Inferno'
# seg_renderer.data_source.data = {col: segments[col].values for col in segments.columns}
# label axes
p.xaxis.axis_label = xvar
p.yaxis.axis_label = yvar

# Plot synapses
lc_legend_added = False
aotu_legend_added = False
tutu_legend_added = False

for bodyId in curr_lc10a_ids:
    # Plot LC10a syn
    df = lc10a_syn_zoom[lc10a_syn_zoom['bodyId']==bodyId]
    lc_kwargs = {}
    if not lc_legend_added:
        lc_kwargs['legend_label'] = 'LC10a'
    p.scatter(df[f'{xvar}'], df[f'{yvar}'], 
                color=lc10a_color, 
                size=markersize, alpha=0.5,
                **lc_kwargs)
    lc_legend_added = lc_legend_added or bool(lc_kwargs)

    # Plot AOTU42 synapses
    df = aotu42_lc10a_syn[aotu42_lc10a_syn['bodyId_post']==bodyId]
    aotu_kwargs = {}
    if not aotu_legend_added:
        aotu_kwargs['legend_label'] = 'AOTU42'
    p.scatter(df[f'{xvar}_post'], df[f'{yvar}_post'], 
                color=aotu42_color, 
                size=markersize, alpha=0.5,
                marker='x',
                **aotu_kwargs)
    aotu_legend_added = aotu_legend_added or bool(aotu_kwargs)

    # Plot TuTuA_2
    df = tutuA2_lc10a_syn[tutuA2_lc10a_syn['bodyId_post']==bodyId]
    tutu_kwargs = {}
    if not tutu_legend_added:
        tutu_kwargs['legend_label'] = 'TuTuA_2'
    p.scatter(df[f'{xvar}_post'], df[f'{yvar}_post'], 
                color=tutu_color, 
                size=10, alpha=0.5,
                marker='x',
                **tutu_kwargs)
    tutu_legend_added = tutu_legend_added or bool(tutu_kwargs)


if p.legend:
    p.legend.location = 'right'
    p.legend.click_policy = 'hide'


if zoom:
    # zoom into synapses:
    xmin = np.floor(min([tutuA2_lc10a_syn[f'{xvar}_post'].min(), aotu42_lc10a_syn[f'{xvar}_post'].min()]))
    xmax = np.ceil(max([tutuA2_lc10a_syn[f'{xvar}_post'].max(), aotu42_lc10a_syn[f'{xvar}_post'].max()]))
    ymin = np.floor(min([tutuA2_lc10a_syn[f'{yvar}_post'].min(), aotu42_lc10a_syn[f'{yvar}_post'].min()]))
    ymax = np.ceil(max([tutuA2_lc10a_syn[f'{yvar}_post'].max(), aotu42_lc10a_syn[f'{yvar}_post'].max()]))
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    p.x_range.start = xlim[0]
    p.x_range.end = xlim[1]
    p.y_range.start = ylim[1]
    p.y_range.end = ylim[0]

p = bokeh_plot_settings(p, min_fontsize=min_fontsize)
# show(p)
   # ensure the output backend supports SVG if using export_svgs
try:
    #p.output_backend = "svg"
    #export_svgs(p, filename=os.path.join(output_dir, "multi_input_synapses.svg"))
    p.output_backend = "canvas"
    export_png(p, filename=os.path.join(output_dir, "compare_synapses.png"))
except Exception as e:
    print(f"Error exporting SVG: {e}")
    output_file(os.path.join(output_dir, "compare_synapses.html"))
    save(p)
#export_png(p, filename="multi_input_synapses.png")

#show(p)

#%%
# try matplotlib
fig, ax = plot_skeleton_matplotlib(example_segments, sorted_lc10a_ids, palette=hue_palette, figsize=(8, 6))

# Add synapses
for bodyId in curr_lc10a_ids:
    df = lc10a_syn_zoom[lc10a_syn_zoom['bodyId']==bodyId]
    ax.scatter(df[f'{xvar}'], df[f'{yvar}'], 
                color=lc10a_color,
                s=5, alpha=0.5)
    df = aotu42_lc10a_syn[aotu42_lc10a_syn['bodyId_post']==bodyId]
    sns.scatterplot( data=df, x=f'{xvar}_post', y=f'{yvar}_post', 
                     color=aotu42_color, marker='x',
                     s=20, alpha=0.7)
    df = tutuA2_lc10a_syn[tutuA2_lc10a_syn['bodyId_post']==bodyId]  
    sns.scatterplot( data=df, x=f'{xvar}_post', y=f'{yvar}_post', 
                     color=tutu_color, marker='x',
                     s=20, alpha=0.7)
# Add custom legend
legend_elements = [Line2D([0], [0], marker='o', color=lc10a_color, lw=0, markersize=3, label='LC10a'),
                    Line2D([0], [0], marker='x', color=aotu42_color, lw=0, markersize=3, label='AOTU042'),
                   Line2D([0], [0], marker='x', color=tutu_color, lw=0, markersize=3, label='TuTuA_2')]
ax.legend(handles=legend_elements, frameon=False, markerscale=2,
          loc='lower right', bbox_to_anchor=(1, 0))
ax.set_title("Compare synapses", loc='left', fontsize=10)

putil.label_figure(fig, figid)
figname = f'skel_compare_example_synapses_{xvar}-{yvar}'
plt.savefig(os.path.join(output_dir, f"{figname}.png"), dpi=300)
print(figname)

# Zoom into AOTU
aotu_lims = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]
xmin, xmax = aotu_lims[xvar].min(), aotu_lims[xvar].max()
ymin, ymax = aotu_lims[yvar].min(), aotu_lims[yvar].max()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.invert_yaxis()

figname = f'skel_compare_example_synapses_{xvar}-{yvar}_zoom'
plt.savefig(os.path.join(output_dir, f"{figname}.png"), dpi=300)
print(figname)


#%%
aotu42_lc10a_syn_counts = aotu42_lc10a_syn.groupby(['bodyId_post'])['bodyId_post'].count().reset_index(name='count_post')
tutuA2_lc10a_syn_counts = tutuA2_lc10a_syn.groupby(['bodyId_post'])['bodyId_post'].count().reset_index(name='count_post')
lc10a_syn_counts = lc10a_syn.groupby(['bodyId'])['bodyId'].count().reset_index(name='count_post')

print(f'{len(aotu42_lc10a_syn_counts)}/{len(lc10a_syn_counts)} AOTU42 synapses to LC10a')
print(f'{len(tutuA2_lc10a_syn_counts)}/{len(lc10a_syn_counts)} TuTuA_2 synapses to LC10a')
aotu42_only_ids = [a for a in aotu42_lc10a_syn['bodyId_post'].unique() if a not in tutuA2_lc10a_syn['bodyId_post'].unique()]
print(f"{len(aotu42_only_ids)} LC10a neurons only receive AOTU42 synapses")


# Which LC10a neurons have both AOTU42 and TuTuA_2 synapses?
both_syn_counts = aotu42_lc10a_syn_counts.merge(tutuA2_lc10a_syn_counts, \
                    on='bodyId_post', how='inner')\
                        .rename(columns={'count_post_x': 'count_aotu42', 'count_post_y': 'count_tutuA2'})
        
print(f'{len(both_syn_counts)}/{len(lc10a_syn_counts)} LC10a neurons have both AOTU42 and TuTuA_2 synapses:')
# %%
xvar = 'z'
yvar = 'y'
xvar_post = f'{xvar}_pre'
yvar_post = f'{yvar}_pre'
both_ids = both_syn_counts['bodyId_post'].unique()

# zoom in to AOTU
aotu_lims = lc10a_syn[lc10a_syn['roi'].str.contains('AOTU')]
xmin, xmax = aotu_lims[xvar].min(), aotu_lims[xvar].max()
ymin, ymax = aotu_lims[yvar].min(), aotu_lims[yvar].max()
#xmin, xmax = (15000, 22000)
#ymin, ymax = (12000, 20000)

fig, axn = plt.subplots(1, 2, figsize=(10, 5))
ax=axn[0]
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color=lc10a_color,
                s=5, alpha=0.5)
sns.scatterplot(data=tutuA2_lc10a_syn, ax=ax,
                x=xvar_post, y=yvar_post,
                color=tutu_color,
                s=5, alpha=0.5, label='TuTuA_2')
sns.scatterplot(data=aotu42_lc10a_syn, ax=ax,
                x=xvar_post, y=yvar_post,
                color=aotu42_color,
                s=5, alpha=0.5, label='AOTU042')
ax.set_title("All LC10a synapses", loc='left', fontsize=10)
ax.legend(frameon=False, markerscale=2,
          loc='lower right', bbox_to_anchor=(1, 0))

ax=axn[1]
sns.scatterplot(data=lc10a_syn, ax=ax,
                x=xvar, y=yvar,
                color=lc10a_color,
                s=5, alpha=0.3)
sns.scatterplot(data=lc10a_syn[lc10a_syn['bodyId'].isin(both_ids)], ax=ax,
                x=xvar, y=yvar,
                color='darkgray',
                s=5, alpha=0.8)
sns.scatterplot(data=tutuA2_lc10a_syn[tutuA2_lc10a_syn['bodyId_post'].isin(both_ids)], ax=ax,
                x=xvar_post, y=yvar_post,
                color=tutu_color,
                s=5, alpha=0.8)
sns.scatterplot(data=aotu42_lc10a_syn[aotu42_lc10a_syn['bodyId_post'].isin(both_ids)], ax=ax,
                x=xvar_post, y=yvar_post,
                color=aotu42_color,
                s=5, alpha=0.8)
ax.set_title('LC10a w/ both AOTU42 and TuTuA_2',
             loc='left', fontsize=10)

for ax in axn:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.invert_yaxis()

putil.label_figure(fig, figid)
figname = f'compare_synapses_{xvar}-{yvar}_both'
plt.savefig(os.path.join(output_dir, f"{figname}.png"), dpi=300)
print(figname)
# %%
import matplotlib as mpl

both_syn_counts['ratio'] = both_syn_counts['count_aotu42'] / both_syn_counts['count_tutuA2']
# Add x, y, z position from lc10a_syn:
both_syn_counts['x'] = [lc10a_syn_zoom[lc10a_syn_zoom['bodyId']==bodyId]['x'].values[0] for bodyId in both_ids]
both_syn_counts['y'] = [lc10a_syn_zoom[lc10a_syn_zoom['bodyId']==bodyId]['y'].values[0] for bodyId in both_ids]
both_syn_counts['z'] = [lc10a_syn_zoom[lc10a_syn_zoom['bodyId']==bodyId]['z'].values[0] for bodyId in both_ids]

# Two-color opponent map, centered at 1:
vmin, vmax = both_syn_counts['ratio'].min(), both_syn_counts['ratio'].max()
ratio_norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=1)
ratio_cmap = 'PiYG'

# Plot ratio vs. position
fig, ax = plt.subplots(figsize=(5, 5))
# Plot lc10a synapses
sns.scatterplot(data=lc10a_syn_zoom, ax=ax,
                x=xvar, y=yvar,
                color=lc10a_color,
                s=5, alpha=0.5)
# Plot highlight synapses
sns.scatterplot(data=both_syn_counts, ax=ax,
                    x=xvar, y=yvar, 
                    hue='ratio', alpha=1,
                    palette=ratio_cmap, 
                    hue_norm=ratio_norm, legend=0)
ax.set_title('Ratio of AOTU42 to TuTuA_2 synapses vs. position', fontsize=10)
ax.set_xlabel(xvar)
ax.set_ylabel(yvar)
ax.set_aspect('equal')
ax.invert_yaxis()

# Custom colorbar
cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.3])
sm = mpl.cm.ScalarMappable(cmap=ratio_cmap, norm=ratio_norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('AOTU42:TuTuA_2', fontsize=10)

 
# %%

# Calculate distances between LC10a synpses using 3d coordinates    :
def compute_synapse_distances_3d(syn_df, x='x', y='y', z='z'):
    syn_df['distance'] = np.sqrt((syn_df[x] - syn_df[x].shift(1))**2 + (syn_df[y] - syn_df[y].shift(1))**2 + (syn_df[z] - syn_df[z].shift(1))**2)
    return syn_df['distance']

# Calculate distance betweeen _pre and _post synapses
def compute_pre_post_distances_3d(syn_df, x_pre='x_pre', y_pre='y_pre', z_pre='z_pre', x_post='x_post', y_post='y_post', z_post='z_post'):
    syn_df['distance'] = np.sqrt((syn_df[x_pre] - syn_df[x_post])**2 + (syn_df[y_pre] - syn_df[y_post])**2 + (syn_df[z_pre] - syn_df[z_post])**2)
    return syn_df['distance']

syn_distances_3d = compute_pre_post_distances_3d(tutuA2_lc10a_syn)
#syn_distances_3d = compute_synapse_distances_3d(tutuA2_lc10a_syn)
#print(f"Minimum distance between LC10a synapses: {syn_distances_3d[syn_distances_3d>1000].min():.1f}")
print(f"Minimum distance between LC10a synapses: {syn_distances_3d.min():.1f}")
print(f"Maximum distance between LC10a synapses: {syn_distances_3d.max():.1f}")
print(f"Mean distance between LC10a synapses: {syn_distances_3d.mean():.1f}")
print(f"Median distance between LC10a synapses: {syn_distances_3d.median():.1f}")
print(f"Standard deviation of distance between LC10a synapses: {syn_distances_3d.std():.1f}")
#%%
# Find spatial overlaps between the two input synapse populations
overlap_radius = np.floor(syn_distances_3d.min()) #13 #500
overlap_df = compute_synapse_overlaps(
    aotu42_lc10a_syn[aotu42_lc10a_syn['bodyId_post'].isin(both_ids)],
    tutuA2_lc10a_syn[tutuA2_lc10a_syn['bodyId_post'].isin(both_ids)],
    radius=overlap_radius,
    xcol=xvar_post, ycol=yvar_post, zcol='z_pre')
print(f"Synapses within {overlap_radius} units: {len(overlap_df)} pairs across "
      f"{overlap_df['bodyId_post'].nunique()} shared neurons.")
if not overlap_df.empty:
    overlap_stats = overlap_df.groupby('bodyId_post').agg(
        pairs=('distance', 'count'),
        min_dist=('distance', 'min')).reset_index()
    avg_pairs = overlap_stats['pairs'].mean()
    print(f"Average overlapping pairs per neuron: {avg_pairs:.1f}; "
          f"closest pair per neuron (median): {overlap_stats['min_dist'].median():.1f}")

#%%
fig, axn = plt.subplots(1, 2, figsize=(10, 5), 
                        sharex=True, sharey=True)
if not overlap_df.empty:
    ax=axn[0]
    ax.scatter(overlap_df['x_aotu'], overlap_df['y_aotu'],
               facecolors='none', edgecolors='black', s=30, linewidth=0.5)
    ax.scatter(overlap_df['x_tutu'], overlap_df['y_tutu'],
               facecolors='none', edgecolors='black', s=30, linewidth=0.5)
    # Plot the corresponding neuron segments
    curr_segments = lc10a_segments[lc10a_segments['bodyId'].isin(overlap_df['bodyId_post'].unique())]
    plot_skeleton_matplotlib(curr_segments, overlap_df['bodyId_post'].unique(), ax=ax)

    # zoom in to the aotu area
    xmin, xmax = lc10a_syn_zoom[xvar].min(), lc10a_syn_zoom[xvar].max()
    ymin, ymax = lc10a_syn_zoom[yvar].min(), lc10a_syn_zoom[yvar].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
ax.set_title('Neurons with overlapping AOTU42/TuTuA_2', fontsize=10)

ax=axn[1]
if not overlap_df.empty:
    ax.set_title(f'Overlapping synapses within {overlap_radius}', fontsize=10)
    sns.scatterplot(data=lc10a_syn_zoom, ax=ax,
                    x=xvar, y=yvar,
                    color=lc10a_color,
                    s=5, alpha=0.5)
    for _, row in overlap_df.iterrows():
        ax.plot([row['x_aotu'], row['x_tutu']],
                [row['y_aotu'], row['y_tutu']],
                color='black', alpha=0.3, linewidth=0.5)
    sns.scatterplot(data=overlap_df, ax=ax,
                    x='x_aotu', y='y_aotu',
                    color=aotu42_color, marker='x',
                    s=30, label='AOTU042')
    sns.scatterplot(data=overlap_df, ax=ax,
                    x='x_tutu', y='y_tutu',
                    color=tutu_color, marker='x',
                    s=30, label='TuTuA_2')
    ax.legend(frameon=False, markerscale=1.5, loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

for ax in axn:
    ax.set_aspect('equal')
ax.invert_yaxis()

# save
putil.label_figure(fig, figid)
figname = f'overlapping_{xvar}-{yvar}'
plt.savefig(os.path.join(output_dir, f"{figname}.png"), dpi=300)
print(figname)


# %%
# All AOTU042 inputs?
# ------------------------------------------------------------
AOTU_inputs_neuron_df, AOTU_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                                   targets=NC(type='AOTU042'),
                                                                   min_total_weight=10)
AOTU_inputs_conn_df = neu.merge_neuron_properties(AOTU_inputs_neuron_df, AOTU_inputs_conn_df, ['type', 'instance'])

#%
# Sort AOTU042 inptus by weight
AOTU_inputs_sorted = AOTU_inputs_conn_df.groupby(['type_pre'])['weight']\
                            .sum().reset_index().sort_values(by='weight', \
                                ascending=False).reset_index()
print("Top 20 AOTU042 input types:")
print(AOTU_inputs_sorted.head(20))

P1_to_AOTU042_types = [i for i in AOTU_inputs_sorted['type_pre'] if 'P1_' in i]
print("P1 types that target AOTU042:")
print(P1_to_AOTU042_types)

#%%
# All TuTuA_2 inputs?
TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                    targets=NC(type='TuTuA_2'),
                                                    min_total_weight=5)
TuTuA2_inputs_conn_df = neu.merge_neuron_properties(TuTuA2_inputs_neuron_df, 
                                                    TuTuA2_inputs_conn_df, 
                                                    ['type', 'instance'])
#%
# Sort TuTuA_2 inptus by weight
TuTuA2_inputs_sorted = TuTuA2_inputs_conn_df.groupby(['type_pre'])['weight']\
                            .sum().reset_index().sort_values(by='weight', \
                                ascending=False).reset_index()
print("Top 20 TuTuA_2 inputs:")
print(TuTuA2_inputs_sorted.head(20))

# P1 types
P1_to_TuTuA2_types = [i for i in TuTuA2_inputs_sorted['type_pre'] if 'P1_' in i]
print("P1 types that target TuTuA_2:")
print(P1_to_TuTuA2_types)
# %%
# All LC10a inputs?
LC10a_inputs_neuron_df, LC10a_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                                   targets=NC(type='LC10a'),
                                                                   min_total_weight=10)
LC10a_inputs_conn_df = neu.merge_neuron_properties(LC10a_inputs_neuron_df, LC10a_inputs_conn_df, ['type', 'instance'])

# Sort inputs by weight
LC10a_inputs_sorted = LC10a_inputs_conn_df.groupby(['type_pre'])['weight']\
                            .sum().reset_index().sort_values(by='weight', \
                                ascending=False).reset_index()
print("Top 20 LC10a input types:")
print(LC10a_inputs_sorted.head(20))

P1_to_LC10a_types = [i for i in LC10a_inputs_sorted['type_pre'] if 'P1_' in i]
print("P1 types that target LC10a:")
print(P1_to_LC10a_types)
