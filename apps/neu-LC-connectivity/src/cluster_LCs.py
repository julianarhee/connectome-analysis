#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : cluster_LCs.py
Created        : 2025/11/12 10:54:10
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/neu-LC-connectivity/src
Author         : jyr
Last Modified  : 
'''
#%%
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas._libs.tslibs.timestamps import integer_op_not_supported
import seaborn as sns

import neuprint as neu
from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import utils as util
import plotting as putil
import neuprint_funcs as npf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
#%%
import importlib
dataset = 'male-cns:v0.9'
c = npf.get_neuprint_client(dataset=dataset)
version = c.fetch_version()
figid = f'{dataset}_{version}'
print(figid)

#%% Plot style
plot_style = 'dark'
putil.set_sns_style(style=plot_style, min_fontsize=16)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%% Output dir
rootdir = '/Volumes/Juliana/connectome'
output_dir = os.path.join(rootdir, 'analyses', 'neuprint', 'cluster_LCs')
processed_dir = os.path.join(rootdir, 'analyses', 'processed_data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Output directory: {output_dir}')

#%%
import importlib
importlib.reload(npf)

# Plot connection matrix between all P1 and all LC
# ------------------------------------------------------------
def get_conn_from_pre_to_post(pre_type, post_type, 
                              client=c, weight_var='percent_of_total'):
    # Get ALL connections between all P1 types and all LC types
    neuron_df, conn_df = neu.fetch_adjacencies(sources=NC(type=f'{pre_type}.*', client=c),
                                            targets=NC(type=f'{post_type}.*', client=c))
    conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])

    # Extract side info
    conn_df = npf.extract_side_from_conn_df(conn_df)

    # Group across side
    conn_df = conn_df.groupby(['type_pre', 'type_post'], \
                                    as_index=False)['weight'].sum()
    #%
    # Normalize by total inputs to each target type
    conn_df = npf.norm_by_specified_inputs(conn_df, group_col='type_post')

    #%
    # Convert to connection matrix
    conn_matrix = connection_table_to_matrix(conn_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
  
    return conn_matrix

#%% 
all_LCs = npf.get_all_LCs(client=c)

#%%
# LC <-> P1 
# ------------------------------------------------------------
pre_type = 'LC'; post_type = 'P1';
weight_var = 'percent_of_total' # 'weight'
use_log = weight_var=='weight'
min_total_weight = 5

#P1_to_LC_conn_matrix = npf.get_conn_from_pre_to_post(pre_type, post_type, client=c, 
#                                                 weight_var=weight_var)
#figname = f'{pre_type}_to_{post_type}_conn_matrix_{weight_var}'
#figsize=(6, 6)
# Get ALL P1 inputs
P1_input_conn_df, P1_input_matrix = npf.get_conn_all_inputs('P1', client=c, 
                                                        return_both=True, 
                                                        weight_var=weight_var,
                                                        min_total_weight=min_total_weight)
#%
# Get all P1 inputs that are LCs
all_sources = all_LCs
LCs_to_P1_neurons = [p for p in P1_input_conn_df['type_pre'].unique() if p in all_LCs] 
LC_to_P1_conn_matrix = P1_input_matrix.loc[LCs_to_P1_neurons].copy()
print(f"Number of LC-P1 neurons: {len(LCs_to_P1_neurons)}")

# Sort by P1 name
P1_names = sorted(LC_to_P1_conn_matrix.columns.tolist(), key=util.natsort)
LC_to_P1_conn_matrix = LC_to_P1_conn_matrix.loc[:, P1_names]

if use_log:
    LC_to_P1_conn = util.log_weights(LC_to_P1_conn_matrix)
    colorbar_label = f'log({weight_var})'
else:
    LC_to_P1_conn = LC_to_P1_conn_matrix
    colorbar_label = weight_var.replace('_', ' ')
#%%
# LC -> P1: plot connection matrix 
figname = f'{pre_type}_to_{post_type}_conn_matrix_{weight_var}_min-{min_total_weight}'
figsize=(12,  5)
vmin=None; vmax=None; 
fig, ax = plt.subplots(figsize=figsize)
npf.plot_connection_matrix(LC_to_P1_conn, ax=ax,
                       vmin=vmin, vmax=0.001,
                       colorbar_label=colorbar_label, cbar_shrink=0.5,
                       normalize_colors=False,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True,
                       grid_color='k', grid_lw=0.1, min_fontsize=12)
# Explicitly enable ticks - seaborn dark style may hide them
ax.tick_params(axis='both', which='major', length=2, width=0.5, 
               color='w', direction='out', bottom=True, top=False, 
               left=True, right=False, pad=2)
plt.subplots_adjust(bottom=0.2)

ax.set_title('{} -> {} direction connections'.format(pre_type, post_type))
ax.set_xlabel('Post-synaptic {} type'.format(post_type))
ax.set_ylabel('Pre-synaptic {} type'.format(pre_type))

# Save
putil.label_figure(fig, figid)
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))
print(figname)

#%%
# LC -> P1: cluster based on cosine similarity
LC_to_P1_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(
                                                    LC_to_P1_conn, 
                                                    method='ward', threshold_percentile=None)
#%
# Plot clustered matrix
fig = npf.plot_connection_matrix(LC_to_P1_clustered,
                       vmin=vmin, vmax=0.1, #vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True, #post_type=='LC',
                       show_all_row_labels=True, show_grid=False,
                       grid_color='w', grid_lw=0.001, min_fontsize=12)
fig.suptitle(f'{pre_type} -> {post_type} connections (cos clustered)'.format(pre_type, post_type))

putil.label_figure(fig, figid)
figname = f'{pre_type}_to_{post_type}_cosine_clustered_{weight_var}_min-{min_total_weight}'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))




# %%
# Get ALL LC OUTPUTS:
# ------------------------------------------------------------
import pickle as pkl
weight_var = 'percent_of_total'
use_log = weight_var=='weight'
create_new = False

#LC_normed_outputs_fpath = os.path.join(output_dir, f'LC_normed_outputs.pkl')

#%%
# 
# neuron_df, conn_df = neu.fetch_adjacencies(sources=NC(type=all_LCs), 
#                                             targets=None,
#                                             client=c,
#                                             min_total_weight=10)
#     
# conn_df = npf.merge_properties_and_group(neuron_df, conn_df)    
# #%%    
# # Normalize all targets by THEIR total inputs
# #if weight_var != 'percent_of_total':
# #if norm_by_all_other_inputs:
# #conn_df = npf.get_and_norm_by_total_inputs(conn_df, c, 
# #                        groupby_type=True, 
# #                        normalize_group_col='type_post')$
# src_types = conn_df['type_pre'].unique() # source types
# target_types = conn_df['type_post'].unique()
# target_types 
# inputs_to_target_neurons, inputs_to_target_conns = neu.fetch_adjacencies(
#                                         sources=None,
#                                         targets=NC(type=target_types),
#                                         client=c, min_total_weight=10)
# #%
# inputs_to_target_conns = neu.merge_neuron_properties(inputs_to_target_neurons, 
#                                     inputs_to_target_conns, ['type', 'instance'])
# #%%
# # Add roi_noside if in groupby_cols:
# inputs_to_target_conns = npf.add_roi_noside(inputs_to_target_conns)
# #%%
# groupby_cols=['type_pre', 'type_post', 'roi_noside']
# inputs_to_target_conns = inputs_to_target_conns.groupby(groupby_cols, \
#                                                             as_index=False)['weight'].sum() 
# #%%
# # Normalize by total inputs to each target type
# inputs_to_target_conns = npf.norm_by_specified_inputs(inputs_to_target_conns, 
#                                                            group_col='type_post')
# #%%
# # Select subset of connections between source and target types
# normalized_output_conn_df = inputs_to_target_conns[\
#                                         inputs_to_target_conns['type_pre'].isin(src_types)].copy()
# 
# conn_matrix = connection_table_to_matrix(normalized_output_conn_df,
#                                     weight_col=weight_var,
#                                     group_cols=['type_pre', 'type_post'],
#                                     sort_by= ['type_pre', 'type_post'])


#%%
#%%
LC_normed_outputs_fpath = npf.get_normed_filepath(processed_dir=processed_dir, neuron_type='LC', io_type='outputs')
LC_output_conn_df, LC_output_matrix = npf.load_normed_data(LC_normed_outputs_fpath, 
                                                     create_new=create_new)
#%%
#%%%
# Get ALL LC INPUTS:
# ------------------------------------------------------------

LC_normed_inputs_fpath = npf.get_normed_filepath(processed_dir=processed_dir, neuron_type='LC', io_type='inputs')
LC_input_conn_df, LC_input_matrix = npf.load_normed_data(LC_normed_inputs_fpath, 
                                                     create_new=create_new)
# try: 
#     with open(LC_normed_inputs_fpath, 'rb') as f:
#         LC_in_tmp = pkl.load(f)
#     LC_input_conn_df = LC_in_tmp['conn_df']
#     LC_input_matrix = LC_in_tmp['conn_matrix']
#     print(f"Loaded: {LC_normed_inputs_fpath}")
# except FileNotFoundError:
#     LC_input_conn_df, LC_input_matrix = npf.get_conn_all_inputs(all_LCs, client=c, 
#                                                                 return_both=True, 
#                                                             weight_var='percent_of_total')
#     LC_in_tmp = {'conn_df': LC_input_conn_df, 'conn_matrix': LC_input_matrix}
#     with open(LC_normed_inputs_fpath, 'wb') as f:
#         pkl.dump(LC_in_tmp, f)
#     print(f"Saved: {LC_normed_inputs_fpath}")
 #

#%%
# Plot ALL LC inputs and outputs
if use_log:
    LC_in = util.log_weights(LC_input_matrix)
    LC_out = util.log_weights(LC_output_matrix)
    colorbar_label = f'log({weight_var})'
else:
    LC_in = LC_input_matrix
    LC_out = LC_output_matrix
    colorbar_label = weight_var

# cluster LC_in and LC_out:
LC_in_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(
                                                    LC_in, 
                                                    method='ward', threshold_percentile=None)
LC_out_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(
                                                    LC_out, 
                                                    method='ward', threshold_percentile=None)
fig, axn = plt.subplots(1, 2, figsize=(10, 5))
for i, mat in enumerate([LC_in_clustered, LC_out_clustered]):
    npf.plot_connection_matrix(mat, ax=axn[i],
                               vmin=vmin, vmax=0.2, #max,
                               colorbar_label=colorbar_label,
                               normalize_colors=True,
                               show_all_col_labels=True, col_label_interval=5 if i==1 else None,
                               show_all_row_labels=i==1, 
                               show_grid=False, grid_color='k', )
    axn[i].set_xlabel('Post-synaptic type')
    axn[i].set_ylabel('Pre-synaptic type')
plt.subplots_adjust(wspace=0.5)

fig.suptitle('LC inputs and outputs')    
# Save
putil.label_figure(fig, figid)
figname = f'LC_inputs_and_outputs_clustered_{weight_var}'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))

#%%
# Sorted matrix
# ------------------------------------------------------------
# Sort ROI name by weight for inputs 
sorted_rois_in = LC_input_conn_df.groupby(['roi_noside'])[weight_var]\
                                    .sum().sort_values(ascending=False).index.tolist() 
# Create a LUT with roi:index mapping
sorted_ix_lut_in = dict((k, i) for i, k in enumerate(sorted_rois_in)) 
# Do same for outputs
sorted_rois_out = LC_output_conn_df.groupby(['roi_noside'])[weight_var]\
                                    .sum().sort_values(ascending=False).index.tolist() 
sorted_ix_lut_out = dict((k, i) for i, k in enumerate(sorted_rois_out)) 

# Sort matrix with ROI:Index LUT
LC_input_sorted = npf.sort_matrix_labels(LC_in, 
                                      conn_df=LC_input_conn_df, 
                                      sort_rows_by='roi_noside',
                                      sort_cols_by='roi_noside',
                                      weight_var=weight_var,
                                      sort_by_lut_row=sorted_ix_lut_in)
LC_output_sorted = npf.sort_matrix_labels(LC_out, 
                                      conn_df=LC_output_conn_df, 
                                      sort_rows_by='roi_noside', #'',
                                      sort_cols_by='roi_noside',
                                      weight_var=weight_var,
                                      sort_by_lut_row=sorted_ix_lut_out)
   
#%%
# PLOT grouped by ROI:
# ROI colors
label_all = True 
sort_by_weights_only = False
pre_variable = 'type_pre'; post_variable = 'type_post';
annot_str = '_labels' if label_all else ''

for cond, mat, conn_df in zip(['inputs', 'outputs'], [LC_input_sorted, LC_output_sorted], [LC_input_conn_df, LC_output_conn_df]):
    pre_grouper = 'roi_noside'; post_grouper = 'roi_noside';
    n_pre_groups = len(conn_df[pre_grouper].unique())
    n_post_groups = len(conn_df[post_grouper].unique())
    pre_grouper_dict = {roi: sns.color_palette("colorblind", n_pre_groups)[i] 
                        for i, roi in enumerate(conn_df[pre_grouper].unique())}
    post_grouper_dict = {roi: sns.color_palette("colorblind", n_post_groups)[i] 
                        for i, roi in enumerate(conn_df[post_grouper].unique())}
    if label_all:
        fig_height = 20 if cond=='inputs' else 12
        fig_width = 12 if cond=='inputs' else 30
    else:
        fig_height = 12
        fig_width = 12
        
    fig = npf.plot_grouped_connection_matrix(mat, conn_df, figsize=(fig_width,fig_height),
                                     pre_grouper_dict=pre_grouper_dict,
                                     post_grouper_dict=post_grouper_dict,
                                     group_per_row=None, group_per_col=None,
                                     pre_grouper = pre_grouper, post_grouper = post_grouper,
                                     sorted_by_grouper=sort_by_weights_only==False,
                                     pre_variable=pre_variable, post_variable=post_variable,
                                     annotate_rows=sort_by_weights_only==False,
                                     annotate_cols=sort_by_weights_only==False,
                                     show_all_row_labels=(label_all or cond=='outputs'),
                                     row_label_interval=4 if cond=='inputs' else None,
                                     show_all_col_labels=(label_all or cond=='inputs'), #cond=='inputs',
                                     col_label_interval=10 if cond=='outputs' else None,
                                     colorbar_label=colorbar_label, min_fontsize=12,
                                     normalize_colors=True, vmin=0, vmax=0.2)
    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(f'LC {cond} grouped')

    putil.label_figure(fig, figid)
    figname = f'LC_{cond}_grouped_{weight_var}{annot_str}'
    print(figname)
    plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))

# %%
# Combine
print(LC_in.shape, LC_out.shape)
# Get intersection of LC_output_matrix index and LC_input_matrix coluns
common_neurons = [r for r in LC_in.columns if r in LC_out.index.tolist()]
print(f"Number of common neurons: {len(common_neurons)}")
inout = pd.concat([LC_input_sorted[common_neurons].T, 
                   LC_output_sorted.loc[common_neurons]], axis=1)
print(inout.shape)
assert inout.shape[1] == LC_in.shape[0] + LC_out.shape[1]
# %%
inout_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(inout, 
                                                                    method='ward')

#%%
# Sort by ROIs?
columns = inout_clustered.columns.tolist()
# Get rois
roi_list = []
for col in columns:
    is_in = col in LC_input_conn_df['type_pre'].unique()
    is_out = col in LC_output_conn_df['type_post'].unique()
    if is_out:
        max_roi = LC_output_conn_df[LC_output_conn_df['type_post']==col]\
                        .groupby('roi_noside')[weight_var].sum()\
                        .sort_values(ascending=False).index[0]
    else:
        max_roi = LC_input_conn_df[LC_input_conn_df['type_pre']==col]\
                        .groupby('roi_noside')[weight_var].sum()\
                        .sort_values(ascending=False).index[0]
    roi_list.append(pd.DataFrame({'type': col, 'roi': max_roi}, index=[0])) 
type_to_roi = pd.concat(roi_list, axis=0).reset_index(drop=True)
# Sort
type_to_roi.sort_values(by='roi', inplace=True)
#type_to_roi
#%
sorted_clustered_cols = type_to_roi['type'].values 

#%%
# Plot clustered matrix
use_transpose=True
if use_transpose:
    mat = inout_clustered.T
else:
    mat = inout_clustered
fig = npf.plot_connection_matrix(inout_clustered.T, #[sorted_clustered_cols],
                       vmin=vmin, vmax=0.5, #None, #1000,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=use_transpose, #False,
                       show_all_row_labels=use_transpose==False, #True, show_grid=False, 
                       grid_color=[0.8]*3, grid_lw=0.001, min_fontsize=12)

ax = fig.axes[0]
ax.set_title('LC input+output combined (cosine similarity clustered)')
if use_transpose:
    ax.set_xlabel('LC type')
    ax.set_ylabel('LC in/out type')
else:
    ax.set_xlabel('LC in/out type')
    ax.set_ylabel('LC type')

putil.label_figure(fig, figid)
figname = 'LC_input_output_connections_clustered'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))

#%%

# Combined in/out conn_df
combo_conn_df = LC_input_conn_df.copy()
combo_conn_df.rename(columns={'type_pre': 'type_post', 'type_post': 'type_pre'}, inplace=True)
combo_conn_df = pd.concat([combo_conn_df, LC_output_conn_df])

#%%
# plot grouped
# ROI colors
pre_grouper = 'roi_noside'; post_grouper = 'roi_noside';
n_pre_groups = len(LC_input_conn_df[pre_grouper].unique())
n_post_groups = len(LC_input_conn_df[post_grouper].unique())
pre_grouper_dict = {roi: sns.color_palette("colorblind", n_pre_groups)[i] 
                    for i, roi in enumerate(LC_input_conn_df[pre_grouper].unique())}
post_grouper_dict = {roi: sns.color_palette("colorblind", n_post_groups)[i] 
                    for i, roi in enumerate(LC_input_conn_df[post_grouper].unique())}

sort_by_weights_only = False
pre_variable = 'type_post'; post_variable = 'type_pre';

fig = npf.plot_grouped_connection_matrix(inout_clustered[sorted_clustered_cols].T,
                                         combo_conn_df, figsize=(20,30),
                                     pre_grouper_dict=pre_grouper_dict,
                                     post_grouper_dict=post_grouper_dict,
                                     group_per_row=None,
                                     group_per_col=None,
                                     pre_grouper = pre_grouper,
                                     post_grouper = post_grouper,
                                     sorted_by_grouper=sort_by_weights_only==False,
                                     pre_variable=pre_variable,
                                     post_variable=post_variable,
                                     annotate_rows=sort_by_weights_only==False,
                                     annotate_cols=sort_by_weights_only==False,
                                     show_all_row_labels=True,
                                     row_label_interval=5,
                                     show_all_col_labels=True, 
                                     col_label_interval=1, #if cond=='outputs' else None,
                                     colorbar_label=colorbar_label,
                                     min_fontsize=12, vmax=0.2)
                                     #



# %%
# Plot dendrograms to understand the clustering
fig_dendro = npf.plot_dendrograms(row_linkage, col_linkage, 
                             row_labels=inout_clustered.index.tolist(),
                             col_labels=inout_clustered.columns.tolist())

# %%




# Start from DNs 
# ------------------------------------------------------------
weight_var = 'percent_of_total'
use_log = weight_var=='weight'
DN_input_conn_df, DN_input_matrix = npf.get_conn_all_inputs('DN', client=c, 
                                                        return_both=True, weight_var=weight_var)
#%%
# Are there type_pre in DN_input that are also type_post in LC_output?
# Intersection of LC_output_conn_df['type_post'] and DN_input_conn_df['type_pre']
LC_DN_1hop_neurons = [l for l in DN_input_conn_df['type_pre'].unique() if l.startswith('LC')]
LC_DN_2hop_neurons = list(set(LC_output_conn_df['type_post'].unique()) & set(DN_input_conn_df['type_pre'].unique()))
print(f"Number of 1-hop LC-DN neurons: {len(LC_DN_1hop_neurons)}")
print(f"Number of 2-hop LC-DN neurons: {len(LC_DN_2hop_neurons)}")
# Number of 1-hop LC-DN neurons: 29
# Number of 2-hop LC-DN neurons: 891

#%% 
# LC-DN 1-hop:  Plot
LC_DN_1hop = DN_input_conn_df[DN_input_conn_df['type_pre'].isin(LC_DN_1hop_neurons)].copy()
LC_DN_1hop_mat = connection_table_to_matrix(LC_DN_1hop,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
colorbar_label = weight_var
curr_vmax = 0.1
fig = npf.plot_connection_matrix(LC_DN_1hop_mat,
                       vmin=vmin, vmax=curr_vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True,
                       grid_color='w', grid_lw=0.005, min_fontsize=12)
fig.suptitle('LC-DN connections (1-hop)')
fig.axes[0].set_xlabel('DN type')
fig.axes[0].set_ylabel('LC type')

putil.label_figure(fig, figid)
figname = f'LC_DN_mat_1hop_{weight_var}'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))

#%%
# Cluster 1hop matrix
LC_DN_1hop_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(LC_DN_1hop_mat, 
                                                                    method='ward')
#%
# Plot clustered matrix
fig = npf.plot_connection_matrix(LC_DN_1hop_clustered,
                       vmin=vmin, vmax=curr_vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=False,
                       grid_color='w', grid_lw=0.005, min_fontsize=12)
fig.suptitle('LC-DN connections (1-hop) (cosine similarity clustered)')
fig.axes[0].set_xlabel('DN type')
fig.axes[0].set_ylabel('LC type')

putil.label_figure(fig, figid)
figname = f'LC_DN_mat_1hop_clustered_{weight_var}'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))


#%%
# 2-hop connections:
# ------------------------------------------------------------

# Get LC targets that are also in the 2-hop LC-DN neurons
LC_to_int = LC_output_conn_df[LC_output_conn_df['type_post'].isin(LC_DN_2hop_neurons)].copy()

# Normalize all targets by total inputs to each target type
LC_to_int_norm_df, LC_to_int_norm_mat = npf.get_conn_all_inputs(None, target_list=LC_to_int['type_post'].unique(), 
                                     client=c, weight_var=weight_var)
#%%
# Get LCs that are in the 2-hop LC-DN neurons
LCs_in_int = LC_to_int['type_pre'].unique()
# Get corresponding matrices
LC_to_int_df = LC_to_int_norm_df[LC_to_int_norm_df['type_pre'].isin(LCs_in_int)].copy()
LC_to_int_mat = LC_to_int_norm_mat.loc[LCs_in_int].copy()

print(LC_to_int_df.shape, LC_to_int_mat.shape)

#%% 
int_to_DN_df = DN_input_conn_df[DN_input_conn_df['type_pre'].isin(LC_DN_2hop_neurons)].copy()

# Get correspond matrices
LC_to_int_mat = connection_table_to_matrix(LC_to_int_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
int_to_DN_mat = connection_table_to_matrix(int_to_DN_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
print(LC_to_int_mat.shape, int_to_DN_mat.shape)

# Multiply the two matrices
LC_DN_2hop_matmul = np.matmul(LC_to_int_mat, int_to_DN_mat)
print(LC_DN_2hop_matmul.shape)

#%%
# Plot DN inputs and outputs
if use_log:
    LC_DN_2hop = util.log_weights(LC_DN_2hop_matmul)
    colorbar_label = f'log({weight_var})'
else:
    LC_DN_2hop = LC_DN_2hop_matmul
    colorbar_label = weight_var

# fig, ax = plt.subplots(figsize=(10, 10))
# fig = npf.plot_connection_matrix(LC_DN_2hop, ax=ax,
#                             vmin=vmin, vmax=vmax,
#                             colorbar_label=colorbar_label,
#                             normalize_colors=True,
#                             show_all_col_labels=True, #i==0,
#                             show_all_row_labels=True, #i==1, 
#                             #show_grid=False, grid_color='k',
#                             )
#%
# Cluster
LC_DN_2hop_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(
                                                                    LC_DN_2hop, 
                                                                    method='ward')
#%%
# Plot clustered matrix
highlight = False
LC_subset = None #['LC10', 'LC16']
label_all = False
annot_str = '_labels' if label_all else ''
min_fontsize_plot = 8 if label_all else 12
colorbar_label = f'{weight_var}' if use_log else weight_var.replace('_', ' ')

if LC_subset is not None:
    # Get subset of LC_DN_2hop.index.tolist() that starts with any of the LC elements in LC_subset
    LC_subset_types = [l for l in LC_DN_2hop.index.tolist() if any(l.startswith(lc) for lc in LC_subset)]
#LC_subset = [l for l in LC_DN_2hop.index.tolist() if l.startswith('LC10')]
if LC_subset is not None:
    LC_subset_str = '-'.join(LC_subset)
    figname = f'LC_DN_2hop_clustered_{weight_var}_subset-{LC_subset_str}'
    figsize = (50, 10)
    col_label_interval = 2
else:
    figname = f'LC_DN_2hop_clustered_{weight_var}'
    figsize = (50, 10) if label_all else (20, 20)
    col_label_interval = 1 if label_all else None
fig = npf.plot_connection_matrix(LC_DN_2hop_clustered, #.loc[LC_subset_types], 
                                 figsize=figsize,
                       vmin=vmin, vmax=0.05, #vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=label_all, 
                       col_label_interval=col_label_interval,
                       show_all_row_labels=True, # Rows are LC type 
                       show_grid=False, grid_color='w', grid_lw=0.005, 
                       min_fontsize=min_fontsize_plot)
fig.suptitle('LC-DN 2hop (cosine similarity clustered)')
fig.axes[0].set_xlabel('DN type')
fig.axes[0].set_ylabel('LC type')

plt.subplots_adjust(left=0.05, right=0.88, bottom=0.2)
# highlight
if highlight:
    DNs_to_highlight = ['DNa03', 'DNa02', 'DNp04']
    LCs_to_highlight = ['LC10a', 'LC6', 'LC9', 'LC16', 'LC12']
    highlight_str = '-'.join(DNs_to_highlight+LCs_to_highlight)
    npf.highlight_row_or_column(fig.axes[0], LC_DN_2hop_clustered, 
                                column_label = DNs_to_highlight,
                                color='r', linewidth=1)
    npf.highlight_row_or_column(fig.axes[0], LC_DN_2hop_clustered, 
                                row_label = LCs_to_highlight,
                                color='r', linewidth=1)
    figname = f'LC_DN_2hop_clustered_{weight_var}_highlight-{highlight_str}{annot_str}'
else:
    figname = f'LC_DN_2hop_clustered_{weight_var}{annot_str}'
print(figname)
putil.label_figure(fig, figid)
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))
# %%

# P1 --> LC connections
# ------------------------------------------------------------
pre_type = 'P1'; post_type = 'LC';
weight_var = 'percent_of_total' # 'weight'
use_log = weight_var=='weight'
min_total_weight = 0

# Get ALL P1 inputs
P1_LC_neuron_df, P1_LC_conn_df = neu.fetch_adjacencies(sources=NC(type=f'{pre_type}_.*'),
                                           targets=NC(type=f'{post_type}.*'),
                                           min_total_weight=min_total_weight)
print(P1_LC_neuron_df.shape, P1_LC_conn_df.shape)
P1_LC_conn_df = npf.merge_properties_and_group(P1_LC_neuron_df, P1_LC_conn_df)

#%%
# conn_df = P1_LC_conn_df.copy()
# 
# src_types = conn_df['type_pre'].unique() # source types
# target_types = conn_df['type_post'].unique()
# src_types 
# #%% 
# # Get all inputs to target types
# inputs_to_target_neurons, inputs_to_target_conns = neu.fetch_adjacencies(
#                                         sources=None,
#                                         targets=NC(type=target_types),
#                                         client=c, min_total_weight=0)
#     
# # Merge columns from neuron_df to conn_df:
# inputs_to_target_conns = neu.merge_neuron_properties(inputs_to_target_neurons, 
#                                     inputs_to_target_conns, ['type', 'instance'])
# #%%
# groupby_cols = ['type_pre', 'type_post', 'roi_noside']
# inputs_to_target_conns = npf.add_roi_noside(inputs_to_target_conns)
# inputs_to_target_conns = inputs_to_target_conns.groupby(groupby_cols, \
#                                                         as_index=False)['weight'].sum()
# #%%
#     # Normalize by total inputs to each target type
# inputs_to_target_conns = npf.norm_by_specified_inputs(inputs_to_target_conns, 
#                                                            group_col='type_post')
#     # Select subset of connections between source and target types
# normalized_output_conn_df = inputs_to_target_conns[\
#                                         inputs_to_target_conns['type_pre'].isin(src_types)].copy()
     
    
    
#%%
# Normalize
P1_LC_conn_df = npf.get_and_norm_by_total_inputs(P1_LC_conn_df, c, 
                                normalize_group_col='type_post',
                                min_total_weight=0,
                                groupby_cols=['type_pre', 'type_post', 'roi_noside'])

# %%
# Connection matrix
P1_LC_conn_matrix = connection_table_to_matrix(P1_LC_conn_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
#%%
# Plot connection matrix
fig = npf.plot_connection_matrix(P1_LC_conn_matrix,
                       vmin=vmin, vmax=0.0005, #None,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True,
                       grid_color='w', grid_lw=0.005, min_fontsize=12)
fig.suptitle(f'{pre_type} to {post_type} connections (normalized by total inputs to each target type)')


putil.label_figure(fig, figid)
figname = f'{pre_type}_to_{post_type}_conn_matrix_{weight_var}_min-{min_total_weight}'
plt.savefig(os.path.join(output_dir, '{}.png'.format(figname)))
# %%
