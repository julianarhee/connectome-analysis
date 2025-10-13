#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Email: juliana.rhee@gmail.com
 # @ Create Time: 2025-03-18 10:53:49
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-03-18 11:24:02
 # @ Description:
 '''
#%%

from fafbseg import flywire
#flywire.set_chunkedgraph_secret("6d9c89da15da56ef5b9b72d4658e5253")

#%%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
#from tqdm.notebook import tqdm
import networkx as nx
#import ray

import time
import pickle as pkl

import utils as util
import network as nw
import importlib

#%%
import plotting as putil

plot_style = 'dark'
putil.set_sns_style(style=plot_style, min_fontsize=6)
bg_color = [0.7]*3 if plot_style == 'dark' else 'k'

#%%
def convert_to_hex_grid(df, flat_topped=True):
    '''
    Convert the x, y coordinates to hex grid coordinates that are interleaved (like retina)
    '''
    if flat_topped: 
        # flat-topped hexagons
        df['x_pq'] = (3/2) * df['p']
        df['y_pq'] = np.sqrt(3) * (df['q'] + df['p'] / 2)
        #df["x_pq"], df["y_pq"] = df["y_pq"], df["x_pq"]
    else:
        # pointy-topped hexagons
        df["x_pq"] = np.sqrt(3) * (df["p"] + df["q"] / 2)
        df["y_pq"] = 1.5 * df["q"]

    theta = np.radians(30)  # or -30
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_new = df["x_pq"]*cos_t - df["y_pq"]*sin_t
    y_new = df["x_pq"]*sin_t + df["y_pq"]*cos_t

    df["x_pq"], df["y_pq"] = x_new, y_new

    return df

def get_CoM_from_pre_post_synapses(pre_to_rows, post_to_rows, LC10a_roots): 
    # Get CoM for LC10A
    c_list = []
    for curr_root in LC10a_roots:
        synapses_pre = pd.DataFrame(pre_to_rows[curr_root], columns=['pre_root_id', 'post_root_id', 'x', 'y', 'z'])  
        synapses_post = pd.DataFrame(post_to_rows[curr_root], columns=['pre_root_id', 'post_root_id', 'x', 'y', 'z'])
        com = util.CoM(synapses_post, xvar='x', yvar='y', zvar='z', is_3d=True) 
        com_pre = util.CoM(synapses_pre, xvar='x', yvar='y', zvar='z', is_3d=True)
        
        cm = pd.DataFrame({'root_id': [curr_root], 'x_den': [com[0]], 'y_den': [com[1]], 'z_den': [com[2]],
                        'x_axo': [com_pre[0]], 'y_axo': [com_pre[1]], 'z_axo': [com_pre[2]]})
        c_list.append(cm)
    coms = pd.concat(c_list)
    
    return coms


def combine_strength_by_column(combined_paths_df, visual_columns):
    '''
    Average/Sum strengths for each PRE neuron corresponding to a column, take both mean and sum.
    NOTE: This does not combine by column, but rather, by PRE neuron, so there can be multiple neurons per column.

    Args:
    combined_paths_df (DataFrame): DataFrame of combined paths
    visual_columns (DataFrame): DataFrame of visual columns (loaded from FlyWire)
    
    Returns:
    column_strengths (DataFrame): visual_columns, with addeed columns (pre, mean_strength, sum_strenght)
    '''
    group_by = 'pre'
    # Average all strengths for VC neurons (pre ids)
    mean_by_col = combined_paths_df.groupby(group_by)['strength'].apply(np.nanmean).reset_index()
    mean_by_col = mean_by_col.rename(columns={'strength': 'mean_strength'})
    # Add all the strengths for VC neurons (pre ids)
    summed_by_col = combined_paths_df.groupby(group_by)['strength'].apply(np.nansum).reset_index()
    summed_by_col = summed_by_col.rename(columns={'strength': 'sum_strength'})
    # Merge
    strength_by_col = pd.merge(mean_by_col, summed_by_col, on=group_by)
    #% convert all column coords
    # Assign column loc to pre_neurons
    column_strengths = visual_columns[visual_columns['root_id'].isin(strength_by_col['pre'])]
    #% Add strength values
    column_strengths['mean_strength'] = [strength_by_col[strength_by_col['pre']==v]['mean_strength'].values[0] for v in column_strengths['root_id']]
    column_strengths['sum_strength'] = [strength_by_col[strength_by_col['pre']==v]['sum_strength'].values[0] for v in column_strengths['root_id']]

    return column_strengths


def plot_hex_grid(visual_columns, df, ax=None,
                  hue_var='strength', hue_min=None, hue_max=None,
                  palette='magma', hex_size=30, hex_edgecolor=[0.3]*3,
                  outline=True, outline_color='w', outline_lw=0.5, lw=0.25):
    '''
    Plot strength values on hex grid, in style of FlyWire visual columns map (https://codex.flywire.ai/app/visual_columns_map)
    
    '''
    if ax is None:
        fig, ax =plt.subplots()
    
    if hue_min is None or hue_max is None:
        # Set hue_norm to the same as the color map
        hue_min = df[hue_var].min() # min is 10?
        hue_max = df[hue_var].max()
    hue_norm = plt.Normalize( hue_min, hue_max )
    
    # Plot base grid    
    sns.scatterplot(x='x_pq', y='y_pq', data=visual_columns, ax=ax,
                    legend=False, marker='H', s=hex_size, 
                    color='none', edgecolor=hex_edgecolor, lw=lw)
    if outline:
        sns.scatterplot(x='x_pq', y='y_pq', data=df, ax=ax,
                        legend=False, marker='H', s=hex_size, 
                        color='none', edgecolor=outline_color, lw=outline_lw)
    # Plot hue 
    sns.scatterplot(x='x_pq', y='y_pq', data=df, hue=hue_var, ax=ax, 
                    palette=palette, hue_norm=hue_norm, legend=False, 
                    marker='H', s=hex_size, edgecolor='none', lw=lw)
    #plt.scatter(df["x_pq"], df["y_pq"], c=df["strength"], cmap="viridis", s=10, ax=ax)
    ax.set_aspect(0.35)

    # plot colorbar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=hue_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label(hue_var)
    # Change last tick label of colorbar to '>50'
    cbar_ticks = cbar.ax.get_yticklabels()#[-1].set_text('>{}'.format(hue_max))
    cbar_ticks[-1].set_text('>{}'.format(hue_max))
    cbar.ax.set_yticklabels(cbar_ticks) #['{}'.format(hue_min), '>{}'.format(hue_max)])

    return ax


#%%
# DATA
skip_connections = True
data_folder = '/Users/julianarhee/Documents/rutalab/connectome_data/FAFB'
processed_data_folder = os.path.join(data_folder, 'processed')

analysis_folder = os.path.join(data_folder, 'analyzed')

files = os.listdir(processed_data_folder)
for f in files:
    if f.endswith('.csv.gz'):
        file_name = f.split('.')[0]
        #if 'proc' in file_name:
        #    file_name = file_name.split('proc_')[-1]
        if skip_connections and 'connections' in file_name:
            continue
        command = file_name+"= pd.read_csv('"+ os.path.join(processed_data_folder, f) +"')"
        exec(command)
        print(command)

# %%
# target connections data
# pre_root_id: root id of the presynaptic neuron, FROM
# post_root_id: root id of the postsynaptic neuron, TO
# neuropil: neuropil abbreviation
# syn_count: number of synapses, aggregated across all connection sites of the given pair and neuopil
# nt_type: neurotransmitter type
# ---------------------------
target_connections = 'connections_princeton_no_threshold'
# target_connections = 'connections'
connections = pd.read_csv(os.path.join(processed_data_folder, target_connections + '.csv.gz'))

#%% 
# Set output dirs
figure_folder = os.path.join(data_folder, 'figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)   


# %% # Get LC10a neurons on RIGHT side
# -------------------------------------
all_LC10 = classification[(classification['clean_cell_type']=='LC10')
                        & (classification['side']=='right')].copy()

# Get Visual Column neurons
visual_columns = util.load_visual_columns_data(data_folder)
visual_columns.head()
VC_roots = visual_columns['root_id'].values

# Only get visual neurons
visual_neuron_types = pd.read_csv(os.path.join(data_folder, 'raw', 'visual_neuron_types.csv.gz'))
visual_neuron_types.head()

# Visual neuropils
#visual_nps = ['AME_R', 'LA_R', 'LO_R', 'LOP_R', 'ME_R']

# Convert column coordinates to hex grid
visual_columns = convert_to_hex_grid(visual_columns, flat_topped=True)

#%%
cell_type = 'LC10a'
LC10a_class = all_LC10[all_LC10['cell_type'] == cell_type].copy()
LC10a_roots = LC10a_class['root_id'].values
print("{} {} neurons in right hemisphere.".format(cell_type, len(LC10a_roots)))
   
#%
connectivity_dir = os.path.join(analysis_folder, '{}_connectivity'.format(cell_type)) 
if not os.path.exists(connectivity_dir):
    os.makedirs(connectivity_dir)

connectivity_figdir = os.path.join(figure_folder, '{}_connectivity'.format(cell_type))
if not os.path.exists(connectivity_figdir):
    os.makedirs(connectivity_figdir)
    
#%%  
importlib.reload(util) 

sort_by_position = True
#%
if sort_by_position:
  
    path_to_synapse_file = os.path.join(data_folder, 'raw', 'synapse_coordinates.csv') 
    syndf = util.load_synapse_coords_df(path_to_synapse_file, LC10a_roots, 
                        filename='{}_synapse_coords.csv'.format(cell_type), 
                        save=True, create_new=False)
    #%    
    lc10_locs = util.get_synapse_loc_each_cell(syndf)
    lc10_locs.groupby('type')['cell'].count()
    #%
    fig, ax = plt.subplots() #1, 2, sharex=True, sharey=True)
    sns.scatterplot(data=lc10_locs, 
                    x='z', y='y', ax=ax, hue='type', s=10)
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_title('{} synapse locations'.format(cell_type))
    
    figname = 'synapse_locations_{}'.format(cell_type)
    plt.savefig(os.path.join(connectivity_figdir, figname + '.png'), dpi=300)

    fig, ax = plt.subplots() #1, 2, sharex=True, sharey=True) 
    sns.scatterplot(data=lc10_locs, #[lc10_locs['type']=='axon'], 
                    x='z', y='y', ax=ax, hue='x', s=10,
                    palette='viridis', legend=0)
    legh = putil.custom_legend(labels=['x'], colors=['yellow'], 
                               use_line=False, markersize=3)
    ax.legend(handles=legh, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.set_aspect(1)
    ax.invert_yaxis()

#% 
# coms = get_CoM_from_pre_post_synapses(pre_to_rows, post_to_rows, LC10a_roots)
# coms.head()

# %%
# ==============================================================================
# Filter connections
# ==============================================================================
pre_neurons = VC_roots #VC_sorted['root_id'].values
post_neurons = LC10a_roots #LC10a_sorted['root_id'].values

# Restrict connections to OL
OL_neuropils = ['ME', 'LOP', 'LO', 'AME', 'LA', 'OCG']
incl_neuropils = ['{}_R'.format(n) for n in OL_neuropils]

min_syn_count = 10
incl_connections = connections[ (connections['neuropil'].isin(incl_neuropils))
                               & (connections['syn_count']>=min_syn_count)]

#%%
# ==============================================================================
# Get paths 
# ==============================================================================
recalculate=False
importlib.reload(nw)

# create the first order connection matrix
matrix_file = os.path.join(connectivity_dir, 'VC_{}_first_order_{}.pkl'.format(cell_type, min_syn_count))
VC_LC10_matrix_first, paths_first = nw.get_first_order(
                                                pre_neurons, 
                                                post_neurons, 
                                                incl_connections, 
                                                matrix_file=matrix_file, 
                                                recalculate=recalculate)
#% create the second order connection matrix
matrix_file = os.path.join(connectivity_dir, 'VC_{}_second_order_syncount{}.pkl'.format(cell_type, min_syn_count))
VC_LC10_matrix_second, paths_second, _ = nw.get_second_order(
                                                pre_neurons, 
                                                post_neurons, incl_connections, 
                                                matrix_file=matrix_file, 
                                                recalculate=recalculate,
                                                return_intermediates=True)  
# #%%
# #% create the third order connection matrix
# matrix_file = os.path.join(connectivity_dir, 'VC_{}_third_order_syncount{}.pkl'.format(cell_type, min_syn_count))
# VC_LC10_matrix_third, paths_third, _ = nw.get_third_order(
#                                                 pre_neurons, post_neurons,
#                                                 incl_connections,
#                                                 matrix_file=matrix_file, 
#                                                 recalculate=recalculate,
#                                                 return_intermediates=True)  
# #% create the fourth order connection matrix
# matrix_file = os.path.join(connectivity_dir, 'VC_{}_fourth_order_syncount{}.pkl'.format(cell_type, min_syn_count))
# VC_LC10_matrix_fourth, paths_fourth, _ = nw.get_fourth_order(
#                                                 pre_neurons, post_neurons, 
#                                                 incl_connections,
#                                                 matrix_file=matrix_file, 
#                                                 recalculate=recalculate,
#                                                 return_intermediates=True)
     
#%
# # Look at connectivity matrix
# for ci, conn_mat in enumerate([VC_LC10_matrix_first, VC_LC10_matrix_second,
#                  VC_LC10_matrix_third, VC_LC10_matrix_fourth]):
#     plt.figure()
#     plt.imshow(conn_mat.T, cmap='hot', interpolation='nearest', aspect='auto')
# 
#     figname = 'conn_matrix_VC_{}_order{}'.format(cell_type, ci+1)
#     plt.savefig(os.path.join(connectivity_figdir, figname))
#     print("Saved: \n{}".format(connectivity_figdir, figname))
     
#%% 
# -------------------------------------
# COMBINE PATHS
# -------------------------------------
use_geom = True
recalculate = True

importlib.reload(nw)
if use_geom:
    combined_paths_file = os.path.join(connectivity_dir, 'VC_{}_syncount{}_combined_paths_geom.pkl'.format(cell_type, min_syn_count))
else:
    combined_paths_file = os.path.join(connectivity_dir, 'VC_{}_syncount{}_combined_paths.pkl'.format(cell_type, min_syn_count))

if os.path.exists(combined_paths_file) and not recalculate:
    try:
        combined_paths_df = pd.read_pickle(combined_paths_file)
    except Exception as e:
        print(e)
        recalculate = True
        print(recalculate)

if recalculate:
    combined_paths_df = nw.combine_paths([paths_first, paths_second], #, paths_third, paths_fourth],
                                     use_geom=use_geom, recalculate=recalculate) 
    # Save combined paths dataframe
    combined_paths_df.to_pickle(combined_paths_file)
    print("Saved combined paths: \n{}".format(combined_paths_file))

#%%
#paths_first['strength'] = paths_first['strength'] ** (1/1)
#paths_second['strength'] = paths_second['strength'] ** (1/2)

#combined_paths_df = pd.concat([paths_first, paths_second], ignore_index=True)
#combined_paths_df = nw.combine_paths([paths_first, paths_second], #paths_third, paths_fourth],
#                                     use_geom=use_geom, recalculate=recalculate) 
#%%
# -------------------------------------------
# Connect each column/ommatidium to LC class
# -------------------------------------------
# Add all strengths for VC neurons (pre ids)

column_strengths = combine_strength_by_column(combined_paths_df, visual_columns)

#%%
n_levels = 2
outline = True
plot_str = '_outline' if outline else ''
hex_edgecolor=[0.3]*3
hex_size = 30
hue_var = 'mean_strength'
palette = 'magma' #'viridis'
#
# Set hue_norm to the same as the color map
hue_min = column_strengths[hue_var].min() # min is 10?
hue_max = column_strengths[hue_var].max()
print(hue_min, hue_max)

#% Convert to hex grid
df = column_strengths.copy()

fig, ax = plt.subplots()
ax = plot_hex_grid(visual_columns, df, ax=ax,
            hue_var=hue_var, hue_min=hue_min, hue_max=hue_max,
            palette=palette, hex_size=hex_size, hex_edgecolor=hex_edgecolor,
            outline=outline, outline_color='w', outline_lw=0.5, lw=0.25)
ax.set_title('{} connections by {} (min={} syn, max depth={})'.format(cell_type, hue_var, min_syn_count, n_levels), loc='left', fontsize=8)

ax.axis('off')

figname = 'VC_{}_{}_syncount{}_N-{}_hex_{}{}'.format(cell_type, hue_var, min_syn_count, n_levels, hue_var, plot_str)
plt.savefig(os.path.join(connectivity_figdir, figname + '.png'), dpi=300)
print(connectivity_figdir, figname)

#%% 
# NOTE: Only going to RIGHT#?

AOTU019_roots = [720575940631517251, 720575940633556644]
AOTU025_roots = [720575940639182424, 720575940616012061]
AOTU_roots = AOTU019_roots + AOTU025_roots
print(AOTU_roots)

AOTU_roots = AOTU025_roots
any_first = connections[(connections['pre_root_id'].isin(paths_first['post']))
                      & (connections['post_root_id'].isin(AOTU_roots))]
any_second = connections[(connections['pre_root_id'].isin(paths_second['post']))
                        & (connections['post_root_id'].isin(AOTU_roots))]
# any_third = connections[(connections['pre_root_id'].isin(paths_third['post']))
#                        & (connections['post_root_id'].isin(AOTU_roots))]
# any_fourth = connections[(connections['pre_root_id'].isin(paths_fourth['post']))
#                         & (connections['post_root_id'].isin(AOTU_roots))]
 
print("First order connections to AOTU: {}".format(any_first.shape[0]))
print("Second order connections to AOTU: {}".format(any_second.shape[0]))
#print("Third order connections to AOTU: {}".format(any_third.shape[0]))
#print("Fourth order connections to AOTU: {}".format(any_fourth.shape[0]))

#%%
# Intermediate -- check which ones end in AOTU
conns_to_aotu19 = connections[(connections['pre_root_id'].isin(combined_paths_df['post']))
                    & (connections['post_root_id'].isin(AOTU019_roots))].copy()
print(conns_to_aotu19.shape)

conns_to_aotu25 = connections[(connections['pre_root_id'].isin(combined_paths_df['post']))
                    & (connections['post_root_id'].isin(AOTU025_roots))].copy()
print(conns_to_aotu25.shape)
#%
VC_AOTU19_roots = conns_to_aotu19['pre_root_id'].unique()
VC_AOTU25_roots = conns_to_aotu25['pre_root_id'].unique()

#% Take subset of combined_paths_df that end in the LCs that project to AOTU:
paths_to_aotu19 = combined_paths_df[combined_paths_df['post'].isin(VC_AOTU19_roots)].copy()
paths_to_aotu25 = combined_paths_df[combined_paths_df['post'].isin(VC_AOTU25_roots)].copy()

print(paths_to_aotu19.shape, paths_to_aotu25.shape, combined_paths_df.shape)

#%%
# Add all strengths for VC neurons (pre ids)
column_strengths_to_aotu19 = combine_strength_by_column(paths_to_aotu19, visual_columns)
column_strengths_to_aotu25 = combine_strength_by_column(paths_to_aotu25, visual_columns)

# summed_by_col_aotu19 = paths_to_aotu19.groupby('pre')['strength'].mean().reset_index()
# summed_by_col_aotu19.head()
# summed_by_col_aotu25 = paths_to_aotu25.groupby('pre')['strength'].mean().reset_index()
# 
# #%
# column_strengths_to_aotu19 = visual_columns[visual_columns['root_id'].isin(summed_by_col_aotu19['pre'])]
# column_strengths_to_aotu25 = visual_columns[visual_columns['root_id'].isin(summed_by_col_aotu25['pre'])]
# print(column_strengths_to_aotu19.shape, column_strengths_to_aotu25.shape, visual_columns.shape)
# 
# #%%
# column_strengths_to_aotu19['strength'] = [summed_by_col_aotu19[summed_by_col_aotu19['pre']==v]['strength'].values[0] for v in column_strengths_to_aotu19['root_id']]
# 
# column_strengths_to_aotu19['log_strength'] = np.log(column_strengths_to_aotu19['strength'])
# column_strengths_to_aotu19['root_strength'] = np.sqrt(column_strengths_to_aotu19['strength'])
# 
# column_strengths_to_aotu25['strength'] = [summed_by_col_aotu25[summed_by_col_aotu25['pre']==v]['strength'].values[0] for v in column_strengths_to_aotu25['root_id']]
# column_strengths_to_aotu25['log_strength'] = np.log(column_strengths_to_aotu25['strength'])
# column_strengths_to_aotu25['root_strength'] = np.sqrt(column_strengths_to_aotu25['strength'])
# 
#column_strengths['log_strength'] = np.log(column_strengths['strength'])
#column_strengths['root_strength'] = np.sqrt(column_strengths['strength'])

print("{}, aotu19 min/min: {:.2f}, {:.2f}".format(hue_var, column_strengths_to_aotu19[hue_var].min(), column_strengths_to_aotu19[hue_var].max()))
print("{}, aotu25 min/min: {:.2f}, {:.2f}".format(hue_var, column_strengths_to_aotu25[hue_var].min(), column_strengths_to_aotu25[hue_var].max()))

#%%
hue_var = 'mean_strength'
palette = 'magma' #'viridis'

# Set hue_norm to the same as the color map
hue_min = 10 #df[hue_var].min() # min is 10?
hue_max = 20 #50 #column_strengths[hue_var].max()
print(hue_min, hue_max)
outline = False

#hue_norm = plt.Normalize( hue_min, hue_max )

hex_edgecolor=[0.3]*3
hex_size = 30

fig, axn = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
for ai, (target, df) in enumerate(zip(['AOTU019', 'AOTU025'], [column_strengths_to_aotu19, column_strengths_to_aotu25])):

    ax=axn[ai]    
    ax = plot_hex_grid(visual_columns, df, ax=ax,
                hue_var=hue_var, hue_min=hue_min, hue_max=hue_max,
                palette=palette, hex_size=hex_size, hex_edgecolor=hex_edgecolor,
                outline=outline, outline_color='w', outline_lw=0.5, lw=0.25)
    ax.set_title('VC->{}->{}'.format(cell_type, target), loc='center', fontsize=8)
    ax.axis('off')
info_str = 'Connections by strength (min={} syn, max depth={})'.format(min_syn_count, n_levels)
fig.text(0.1, 0.95, info_str, fontsize=8)

figname = 'split-AOTU19_AOTU25_VC_{}_syncount{}_N-{}_hex_{}'.format(cell_type, min_syn_count, n_levels, hue_var)
plt.savefig(os.path.join(connectivity_figdir, figname + '.png'), dpi=300)
print(connectivity_figdir, figname)

# %%

# Get actual connections between 1st and 2nd order LC10a neurons

#%%
def connections_df_to_pivot_matrix(filtered_connections, pre_neurons, post_neurons):
    grouped = filtered_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    connection_matrix_df = grouped.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    connection_matrix_df = connection_matrix_df.reindex(index=pre_neurons, columns=post_neurons, fill_value=0)
    
    return connection_matrix_df.values
#%%
def get_valid_connections_by_strength(connections):
    # Filter connections for valid synapse strengths
    valid_connections = connections.copy()
    valid_connections = valid_connections[~np.isnan(valid_connections['syn_strength'])]
    valid_connections = valid_connections[(valid_connections['syn_strength'] > 0) & (valid_connections['syn_strength'] < 1)]

    return valid_connections

def get_connections_to_target_from_first_order_paths(pre_neurons, downstream_neurons, #paths_first,
                                                     target_roots, valid_connections):
    '''
    pre_neurons (list): neurons pre-synaptic to downstream_neurons
    downstream_neurons (list): neurons post-synaptic to pre_neurons and pre-synaptic to target_roots
    target_roots (list): neurons post-synaptic to downstream_neurons
    valid_connections (DataFrame): DataFrame of all connections
    '''
    # pre_neurons = paths_first['pre'].unique()        
    # Neurons that are "downstream" of the previous path (i.e., post neurons from a given path set) 
    #downstream_neurons = to_upstream_neurons # 
    #downstream_neurons = paths_first['post'].unique() #values # #downstream['post_root_id'].unique()
       
    # Determine neurons UPSTREAM to target neurons (which we hope is a subset of downstream_neurons)
    upstream = valid_connections[valid_connections['post_root_id'].isin(target_roots)]
    upstream_neurons = upstream['pre_root_id'].unique()
        
    # Common intermediate neurons between the two groups
    common_neurons = np.intersect1d(downstream_neurons, upstream_neurons)
    #print(len(common_neurons))
    
    # remove any neurons in common_neurons that are also in pre_neurons or post_neurons to prevent cycles
    common_neurons = common_neurons[~np.isin(common_neurons, np.concatenate([pre_neurons, target_roots]))]    
    
    # Build pre->common matrix
    pre_common_connections = connections[
        connections['pre_root_id'].isin(pre_neurons) & 
        connections['post_root_id'].isin(common_neurons)
    ]
    pre_common_group = pre_common_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    pre_common_df = pre_common_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    pre_common_df = pre_common_df.reindex(index=pre_neurons, columns=common_neurons, fill_value=0)
    pre_common_matrix = pre_common_df.values

    # Build common->post matrix
    common_post_connections = connections[
        connections['pre_root_id'].isin(common_neurons) &
        connections['post_root_id'].isin(target_roots)
    ]
    common_post_group = common_post_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    common_post_df = common_post_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    common_post_df = common_post_df.reindex(index=common_neurons, columns=target_roots, fill_value=0)
    common_post_matrix = common_post_df.values

    # Multiply the matrices to get second order connectivity and take geometric mean
    #second_order = np.dot(pre_common_matrix, common_post_matrix)
    #second_order = np.power(second_order, 1/2)

    return common_neurons, pre_common_matrix, common_post_matrix

#%%
valid_connections = get_valid_connections_by_strength(connections)

#%%

#valid_connections[(valid_connections['pre_root_id'].isin(paths_first['post'].unique()))
#                  & (valid_connections['post_root_id'].isin(AOTU019_roots))] 

#%%        
# Get connections from first order to AOTU19
# PRE neurons (VC roots), 
# common_neurons are intersection of post_neurons_first and AOTU roots
# POST neurons are AOTU019
#post_first = np.unique(paths_first['post'].values)
#post_first_and_aotu_common = np.intersect1d(post_first, AOTU019_roots)
common_neurons_19, pre_common_matrix_19, common_post_matrix_19 = get_connections_to_target_from_first_order_paths(
                                                                        paths_first['pre'].unique(), 
                                                                        paths_first['post'].unique(),
                                                                        AOTU019_roots, 
                                                                        valid_connections)
# And corresponding paths:
# This should be same as: 
# paths_to_aotu19[(paths_to_aotu19['I1']==-1) & (paths_to_aotu19['I2']==-1) & (paths_to_aotu19['I3']==-1)]
aotu19_paths_from_first = nw.get_explicit_paths_sparse_second_with_strength(
                                    #pre_neurons,            # array of pre neuron IDs
                                    paths_first['pre'].unique(), 
                                    common_neurons_19,         # array of neurons between pre and post 
                                    AOTU019_roots,           # array of post neuron IDs 
                                    pre_common_matrix_19, # dense matrix: (n_pre, n_lc)
                                    common_post_matrix_19  # dense matrix: shape (n_lc, n_post)
                                    )

# Get connections from first order to AOTU025
common_neurons_25, pre_I1_matrix_25, I1_post_matrix_25 = get_connections_to_target_from_first_order_paths(
                                                                        paths_first['pre'].unique(),
                                                                        paths_first['post'].unique(),
                                                                        AOTU025_roots, 
                                                                        valid_connections)    
# And corresponding paths:
aotu25_paths_from_first = nw.get_explicit_paths_sparse_second_with_strength(
                                    #pre_neurons,            # array of pre neuron IDs
                                    paths_first['pre'].unique(),
                                    common_neurons_25,         # array of neurons between pre and post  
                                    AOTU025_roots,           # array of post neuron IDs 
                                    pre_I1_matrix_25, # dense matrix: (n_pre, n_lc)
                                    I1_post_matrix_25  # dense matrix: shape (n_lc, n_post)
                                    )

#%%   j 
to_19 = paths_to_aotu19[(paths_to_aotu19['I1']!=-1) 
                      & (paths_to_aotu19['I2']==-1) 
                      & (paths_to_aotu19['I3']==-1)]

to_25 = paths_to_aotu25[(paths_to_aotu25['I1']!=-1) 
                      & (paths_to_aotu25['I2']==-1) 
                      & (paths_to_aotu25['I3']==-1)]

# Get conns to AOTU from 2nd order
# -------------------------------------
# get conn matrix for connected VC to "pre" (i.e., I1 for 2nd order)
filtered_conns_19= valid_connections[(valid_connections['pre_root_id'].isin(to_19['pre'].unique()))]
filtered_conns_25 =  valid_connections[(valid_connections['pre_root_id'].isin(to_25['pre'].unique()))]

#pre_I1_matrix_second_connected = connections_df_to_pivot_matrix(filtered_conns,
#                                        paths_second['pre'].unique(), common_neurons_second)
#print(pre_I1_matrix_second_connected.shape)
#%
pre_I1_matrix_second_19 = connections_df_to_pivot_matrix(filtered_conns_19,
                                        to_19['pre'].unique(), to_19['I1'].unique())

pre_I1_matrix_second_25 = connections_df_to_pivot_matrix(filtered_conns_25,
                                        to_25['pre'].unique(), to_25['I1'].unique())
#%%
# From AOTU19
I2_neurons_19, I1_I2_matrix_19, I2_post_matrix_19 = get_connections_to_target_from_first_order_paths(
                                                                        to_19['I1'].unique(), #paths_second['I1'].unique(),
                                                                        to_19['post'].unique(), #paths_second['post'].unique(),
                                                                        AOTU019_roots, 
                                                                        valid_connections)    

# Get connections from first order to AOTU025
I2_neurons_25, I1_I2_matrix_25, I2_post_matrix_25 = get_connections_to_target_from_first_order_paths(
                                                                        to_25['I1'].unique(), #paths_second['I1'].unique(),
                                                                        to_25['post'].unique(), ##paths_second['post'].unique(),
                                                                        AOTU025_roots, 
                                                                        valid_connections)    
#%% 
aotu19_paths_from_second = nw.get_explicit_paths_sparse_third_with_strength(
                                    to_19['pre'].unique(), #paths_second['pre'].unique(),            # e.g., array of pre_neuron IDs
                                    to_19['I1'].unique(), #common_neurons_second, #pre_neurons,       # aka, common_neurons_second, array of I1 neuron IDs
                                    I2_neurons_19, #common_neurons,       # array of I2 neuron IDs
                                    AOTU019_roots,           # array of post neuron IDs
                                    pre_I1_matrix_second_19,    # dense matrix: shape (n_pre, n_I1)
                                    I1_I2_matrix_19, #pre_common_matrix,     # dense matrix: shape (n_I1, n_I2)
                                    I2_post_matrix_19 #common_post_matrix    # dense matrix: shape (n_I2, n_post)
                            )
#%% 
#to_25 = paths_to_aotu25[(paths_to_aotu25['I1']!=-1) & (paths_to_aotu25['I2']==-1) & (paths_to_aotu25['I3']==-1)]
sorted(to_25['post'].unique()) == sorted(I2_neurons_25)

aotu25_paths_from_second = nw.get_explicit_paths_sparse_third_with_strength(
                                    #paths_second['pre'].unique(),            # e.g., array of pre_neuron IDs
                                    to_25['pre'].unique(),
                                    to_25['I1'].unique(), #common_neurons_second, #pre_neurons,       # aka, common_neurons_second, array of I1 neuron IDs
                                    I2_neurons_25, #common_neurons,       # array of I2 neuron IDs
                                    AOTU025_roots,           # array of post neuron IDs
                                    pre_I1_matrix_second_25,    # dense matrix: shape (n_pre, n_I1)
                                    I1_I2_matrix_25, #pre_common_matrix,     # dense matrix: shape (n_I1, n_I2)
                                    I2_post_matrix_25 #common_post_matrix    # dense matrix: shape (n_I2, n_post)
                            )

#%%
use_geom=True
if use_geom:
    aotu19_paths_from_first['strength'] = aotu19_paths_from_first['strength'] ** (1/2)
    aotu19_paths_from_second['strength'] = aotu19_paths_from_second['strength'] ** (1/3)

    aotu25_paths_from_first['strength'] = aotu25_paths_from_first['strength'] ** (1/2)
    aotu25_paths_from_second['strength'] = aotu25_paths_from_second['strength'] ** (1/3)
    
#%%
# Combined AOTU pahts
standard_columns_second_order = ["pre", "I1", "I2", "post", "strength"]
# Standardize each dataframe:
aotu19_paths_first_std  = nw.standardize_paths_df(aotu19_paths_from_first.copy(), standard_columns_second_order)
aotu19_paths_second_std = nw.standardize_paths_df(aotu19_paths_from_second.copy(), standard_columns_second_order)

# Combine paths
paths_to_aotu19_df = pd.concat([ aotu19_paths_first_std, aotu19_paths_second_std], # paths_third_std, paths_fourth_std],
                                ignore_index=True)

#%%
#  Standardize each dataframe:
aotu25_paths_first_std  = nw.standardize_paths_df(aotu19_paths_from_first.copy(), standard_columns_second_order)
aotu25_paths_second_std = nw.standardize_paths_df(aotu25_paths_from_second.copy(), standard_columns_second_order)

# Combine paths
paths_to_aotu25_df = pd.concat([ aotu25_paths_first_std, aotu25_paths_second_std], # paths_third_std, paths_fourth_std],
                                ignore_index=True)
#%%
for p, pdf in paths_to_aotu19_df.groupby('pre'):
    pdf

#%%
column_strengths_to_aotu19 = combine_strength_by_column(paths_to_aotu19_df, visual_columns) 
column_strengths_to_aotu25 = combine_strength_by_column(paths_to_aotu25_df, visual_columns)
#%% 
print(column_strengths_to_aotu19[hue_var].min(), column_strengths_to_aotu19[hue_var].max())
print(column_strengths_to_aotu25[hue_var].min(), column_strengths_to_aotu25[hue_var].max())
#%%

outline = False
plot_str = '_outline' if outline else ''
# Set hue_norm to the same as the color map
hue_var = 'sum_strength'
hue_min = column_strengths_to_aotu19[hue_var].min() # min is 10?
#hue_max = 25 if hue_var == 'mean_strength' else column_strengths_to_aotu19[hue_var].max()
hue_max = column_strengths_to_aotu19[hue_var].max()
palette = 'magma' #'viridis'
print(hue_min, hue_max)
hue_norm = plt.Normalize( hue_min, hue_max )

edgecolor=[0.3]*3
edgecolor2 = [0.4]*3
marker_size = 25


fig, axn = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)

for ai, (target, df) in enumerate(zip(['AOTU019', 'AOTU025'], [column_strengths_to_aotu19, column_strengths_to_aotu25])):
    
    ax=axn[ai]

    ax = plot_hex_grid(visual_columns, df, ax=ax,
                hue_var=hue_var, hue_min=hue_min, hue_max=hue_max,
                palette=palette, hex_size=hex_size, hex_edgecolor=hex_edgecolor,
                outline=outline, outline_color='w', outline_lw=0.5, lw=0.25)
    ax.set_title('VC->{}->{}'.format(cell_type, target), loc='center', fontsize=8)
    
    #df = column_strengths_to_aotu19.copy()

#     sns.scatterplot(x='x_pq', y='y_pq', data=visual_columns, ax=ax,
#                     legend=False, marker='H', s=marker_size, 
#                     color='none', edgecolor=edgecolor, lw=0.25)
#     sns.scatterplot(x='x_pq', y='y_pq', data=df, hue=hue_var, ax=ax, 
#                     palette=palette, hue_norm=hue_norm, legend=False, 
#                     marker='H', s=marker_size, edgecolor=edgecolor, lw=0.25)
#     if outline:
#         sns.scatterplot(x='x_pq', y='y_pq', data=column_strengths, ax=ax,
#                         legend=False, marker='H', s=marker_size, 
#                         color='none', edgecolor=edgecolor2, lw=0.25)
# 
#     #plt.scatter(df["x_pq"], df["y_pq"], c=df["strength"], cmap="viridis", s=10, ax=ax)
#     ax.set_aspect(0.35)
    ax.set_title('VC->{}->{}'.format(cell_type, target), loc='center', fontsize=8)
    ax.axis('off')

info_str = 'Connections by {} (min={} syn, max depth={})'.format(hue_var, min_syn_count, n_levels)
fig.text(0.1, 0.95, info_str, fontsize=8)

figname = 'split-weighted-{}-AOTU19_AOTU25_VC_{}_syncount{}_N-{}_hex_{}{}'.format(hue_var, cell_type, min_syn_count, n_levels, hue_var, plot_str)
plt.savefig(os.path.join(connectivity_figdir, figname + '.png'), dpi=300)
print(connectivity_figdir, figname)



#%%

# make a histogram of visual_neuron_types type
counts_by_type_19 = visual_neuron_types[visual_neuron_types['root_id'].isin(pre_aotu19)]['type'].value_counts().reset_index()
counts_by_type_25 = visual_neuron_types[visual_neuron_types['root_id'].isin(pre_aotu25)]['type'].value_counts().reset_index()

fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)

for ai, (counts_by_type, title) in enumerate(zip([counts_by_type_19, counts_by_type_25], ['AOTU19', 'AOTU25'])):
    ax=axn[ai]
    sns.barplot(data=counts_by_type, x='type', y='count', ax=ax, 
                color=bg_color)
    ax.set_box_aspect(1)
    ax.set_title('VC->{} neuron types'.format(title), fontsize=8)

figname = 'cell_types_AOTU19_AOTU25_VC_{}_syncount{}_N-{}'.format(cell_type, min_syn_count, n_levels)
plt.savefig(os.path.join(figure_folder, figname + '.png'), dpi=300)



#%%#%%#%a
# 
# 
print(column_strengths_to_aotu19.shape, column_strengths_to_aotu25.shape, visual_columns.shape)

# %%
