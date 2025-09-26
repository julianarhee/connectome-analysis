#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : TuTuA_LC19a_recurrent.py
Created        : 2025/09/25 10:41:37
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/TuTu-LC1a0-AOTU/src
Author         : jyr
Last Modified  : 
'''
#%%
import os
import glob
from re import L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib for inline plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotting as putil

interactive_3D = False

if interactive_3D:
    # Try to use interactive backend, fall back to default if not available
    try:
        matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactivity
        print("Using Qt5Agg backend for interactive 3D plots")
    except ImportError:
        try:
            matplotlib.use('TkAgg')  # Fallback to TkAgg
            print("Using TkAgg backend for interactive 3D plots")
        except ImportError:
            print("Using default backend (may not be fully interactive)")
            pass
else:
    # Let matplotlib use its default backend for the current environment
    # In Jupyter notebooks, this is typically 'inline'
    pass

# Check current backend
print(f"Default matplotlib backend: {matplotlib.get_backend()}")
print(f"Running in Jupyter: {'jupyter' in str(type(__builtins__)) or 'IPython' in str(type(__builtins__))}")

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
                         markersize=20, marker='x'):
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
    #ax.invert_yaxis()
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

    ax.invert_yaxis()
    ax.invert_xaxis()

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
    bins_pc2 = np.linspace(tutu_lc10a_syn_pca['PC2'].min(), tutu_lc10a_syn_pca['PC2'].max(), n_bins_pc2)
    tutu_lc10a_syn_pca['PC2_bin'] = pd.cut(tutu_lc10a_syn_pca['PC2'], bins_pc2)
    tutu_lc10a_syn_pca['PC2_bin_label'] = tutu_lc10a_syn_pca['PC2_bin'].apply(lambda x: x.mid)

    # Count how many points fall into each bin
    tutu_lc10a_syn_pca_binned1= tutu_lc10a_syn_pca.groupby(['PC1_bin_label']).agg({'PC1': 'count', 'syn_count': 'sum'}).reset_index()
    # Same to count PC2
    tutu_lc10a_syn_pca_binned2 = tutu_lc10a_syn_pca.groupby(['PC2_bin_label']).agg({'PC2': 'count', 'syn_count': 'sum'}).reset_index()   
    tutu_lc10a_syn_pca_binned = pd.concat([tutu_lc10a_syn_pca_binned1, tutu_lc10a_syn_pca_binned2], axis=1)
    # Combine counts for PC1 and 2

    # Convert PC1_bin_label to numeric to ensure proper x-axis alignment
    tutu_lc10a_syn_pca_binned['PC1_bin_numeric'] = pd.to_numeric(tutu_lc10a_syn_pca_binned['PC1_bin_label'])
    tutu_lc10a_syn_pca_binned['PC2_bin_numeric'] = pd.to_numeric(tutu_lc10a_syn_pca_binned['PC2_bin_label'])

    return tutu_lc10a_syn_pca_binned

def plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, lc10a_cdict,
                          hue_var='post_root_id', bin_cmap = 'viridis_r',
                          markersize=20, marker='x', 
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
                    palette=lc10a_cdict, legend=0)
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


def synapse_matches(tutu_lc10a_syn, lc10a_tutu_syn):
    # Spatial mapping analysis: Check if pre/post coordinates in tutu_lc10a_syn 
    # correspond to post/pre coordinates in lc10a_tutu_syn
    # Find MATCHES: between each pair of TuTu->LC10a and LC10a->TuTu,
    # find the TuTu that connects to an LC10a (FWD), and the same LC10a that connects back to it (FB)

    # This reveals connection patterns:
    # Reciprocal: both directions (most points)
    # Unidirecitonal: points on axes
    # Connection strength correlations: whether strong FF connections correlate with strong FB connections

    # Examine network topology with scatter plot tutu->LC10a vs LC10a->TuTu
    # FF dominance: more points below diagonal
    # FB dominance: more points above
    # Balanced network: points clustered around diagonal


    # First, let's examine the coordinate columns available
    print("tutu_lc10a_syn columns:", tutu_lc10a_syn.columns.tolist())
    print("lc10a_tutu_syn columns:", lc10a_tutu_syn.columns.tolist())

    # Check if we have coordinate data
    coord_cols = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']
    tutu_coords_available = all(col in tutu_lc10a_syn.columns for col in coord_cols)
    lc10a_coords_available = all(col in lc10a_tutu_syn.columns for col in coord_cols)

    print(f"tutu_lc10a_syn has coordinate columns: {tutu_coords_available}")
    print(f"lc10a_tutu_syn has coordinate columns: {lc10a_coords_available}")

    if tutu_coords_available and lc10a_coords_available:
        # Create coordinate pairs for comparison
        # tutu_lc10a_syn: pre -> post
        # lc10a_tutu_syn: pre -> post (but we want to check if this matches post -> pre from tutu_lc10a_syn)
        
        # For each connection in tutu_lc10a_syn, find potential matches in lc10a_tutu_syn
        # where the pre of tutu_lc10a_syn matches the post of lc10a_tutu_syn
        # and the post of tutu_lc10a_syn matches the pre of lc10a_tutu_syn
        
        # Create a function to calculate 3D distance
        def calculate_distance(row1, row2):
            """Calculate 3D Euclidean distance between two coordinate sets"""
            dx = row1['pre_x'] - row2['post_x']
            dy = row1['pre_y'] - row2['post_y'] 
            dz = row1['pre_z'] - row2['post_z']
            return np.sqrt(dx**2 + dy**2 + dz**2)
        
        # For each connection in tutu_lc10a_syn, find the closest lc10a_tutu_syn connection
        # that has matching root IDs (but reversed direction)
        matches = []
        
        for idx, tutu_row in tutu_lc10a_syn.iterrows():
            # Find lc10a_tutu_syn connections where:
            # tutu_row['pre_root_id'] == lc10a_row['post_root_id'] AND
            # tutu_row['post_root_id'] == lc10a_row['pre_root_id']
            matching_connections = lc10a_tutu_syn[
                (lc10a_tutu_syn['post_root_id'] == tutu_row['pre_root_id']) &
                (lc10a_tutu_syn['pre_root_id'] == tutu_row['post_root_id'])
            ]
            
            if len(matching_connections) > 0:
                # Calculate distances to all matching connections
                distances = []
                for _, lc10a_row in matching_connections.iterrows():
                    dist = calculate_distance(tutu_row, lc10a_row)
                    distances.append(dist)
                
                # Find the closest match
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]
                closest_match = matching_connections.iloc[min_dist_idx]
                
                # Calculate total synapses for normalization
                # Use shape[0] to count all individual synapses, not sum of syn_count
                total_tutu_to_lc10a = tutu_lc10a_syn.shape[0]
                total_lc10a_to_tutu = lc10a_tutu_syn.shape[0]
                
                # Calculate normalized synapse counts as fractions
                syn_count_tutu_norm = tutu_row['syn_count'] / total_tutu_to_lc10a
                syn_count_lc10a_norm = closest_match['syn_count'] / total_lc10a_to_tutu
                
                matches.append({
                    'tutu_idx': idx,
                    'lc10a_idx': matching_connections.index[min_dist_idx],
                    'distance': min_distance,
                    'tutu_pre_id': tutu_row['pre_root_id'],
                    'tutu_post_id': tutu_row['post_root_id'],
                    'syn_count_tutu': tutu_row['syn_count'],
                    'syn_count_lc10a': closest_match['syn_count'],
                    'syn_count_tutu_norm': syn_count_tutu_norm,
                    'syn_count_lc10a_norm': syn_count_lc10a_norm
                })
        
        # Convert to DataFrame for analysis
        matches_df = pd.DataFrame(matches)
  
    return matches_df

def hist_synapse_distances(matches_df, verbose=False):
    if verbose:
        print(f"Found {len(matches_df)} potential matches")
        print(f"Distance statistics:")
        print(f"  Mean distance: {matches_df['distance'].mean():.2f}")
        print(f"  Median distance: {matches_df['distance'].median():.2f}")
        print(f"  Min distance: {matches_df['distance'].min():.2f}")
        print(f"  Max distance: {matches_df['distance'].max():.2f}") 

    fig1, ax = plt.subplots(1, 1, figsize=(5, 4)) 
    # Distance histogram
    ax.hist(matches_df['distance'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Spatial Distance (μm)')
    ax.set_ylabel('Number of Matches')
    ax.set_title('Distribution of Spatial Distances')
    ax.axvline(matches_df['distance'].median(), color='red', linestyle='--', 
                label=f'Median: {matches_df["distance"].median():.1f}μm')
    ax.set_box_aspect(1)
    ax.legend()
   
    return fig1

def scatter_synapse_counts(matches_df, lc10a_cdict):
    # Scatter plot: syn_count_tutu vs syn_count_lc10a with color coding by LC10a ID
    # Create color mapping based on LC10a IDs
    # LC10a ID is post_root_id in tutu_lc10a_syn and pre_root_id in lc10a_tutu_syn
    
    fig2, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    colors = []
    gray_count = 0
    colored_count = 0
    
    # Convert the entire column to int64 to fix the data type issue
    matches_df['tutu_post_id'] = matches_df['tutu_post_id'].astype('int64')
    
    # Use direct column access - get the entire column as a numpy array
    tutu_post_ids = matches_df['tutu_post_id'].values  # This preserves the int64 type
    
    for i in range(len(matches_df)):
        # Use the post_root_id from tutu_lc10a_syn (which is the LC10a ID)
        # This should be the same as the pre_root_id from lc10a_tutu_syn
        lc10a_id_np = tutu_post_ids[i]  # Direct array access preserves int64
        # Get color from lc10a_cdict, default to gray if not found
        color = lc10a_cdict.get(lc10a_id_np, 'gray')
        
        if color == 'gray':
            gray_count += 1
        else:
            colored_count += 1
        
        colors.append(color)
    
    ax.scatter(matches_df['syn_count_tutu'], matches_df['syn_count_lc10a'], 
                c=colors, alpha=0.6, s=50)
    ax.set_xlabel('TuTu→LC10a Synapse Count')
    ax.set_ylabel('LC10a→TuTu Synapse Count')
    ax.set_title('Synapse Count Correlation')
    
    # Add diagonal line for perfect correlation
    max_count = max(matches_df['syn_count_tutu'].max(), matches_df['syn_count_lc10a'].max())
    ax.plot([0, max_count], [0, max_count], 'r--', alpha=0.5, label='Perfect correlation')
    ax.set_aspect('equal')
    
    ax.legend()
    
    plt.tight_layout()
    #plt.show()
 
    return fig2

def extract_directional_neurons(matches_df, balance_threshold=0.1, use_normalized=True):
    """
    Extract neurons that show directional bias in synapse counts.
    
    Parameters:
    -----------
    matches_df : pd.DataFrame
        DataFrame with columns 'syn_count_tutu', 'syn_count_lc10a', 'tutu_pre_id', 'tutu_post_id'
        and optionally 'syn_count_tutu_norm', 'syn_count_lc10a_norm'
    balance_threshold : float
        Fraction of max count to consider as "balanced" (default: 0.1 = 10%)
    use_normalized : bool
        Whether to use normalized counts (fractions) instead of raw counts
    
    Returns:
    --------
    dict : Dictionary with keys:
        - 'tutu_dominant': List of TuTu IDs with TuTu->LC10a > LC10a->TuTu
        - 'lc10a_dominant': List of LC10a IDs with LC10a->TuTu > TuTu->LC10a
        - 'balanced': List of (tutu_id, lc10a_id) pairs with similar counts
    """
    
    if len(matches_df) == 0:
        return {
            'tutu_dominant': [],
            'lc10a_dominant': [],
            'balanced': []
        }
    
    # Choose which counts to use
    if use_normalized and 'syn_count_tutu_norm' in matches_df.columns:
        count_tutu = matches_df['syn_count_tutu_norm']
        count_lc10a = matches_df['syn_count_lc10a_norm']
        print("Using normalized synapse counts (fractions of total)")
    else:
        count_tutu = matches_df['syn_count_tutu']
        count_lc10a = matches_df['syn_count_lc10a']
        print("Using raw synapse counts")
    
    # Calculate wiggle room for balanced classification
    max_count = max(count_tutu.max(), count_lc10a.max())
    wiggle_room = balance_threshold * max_count
    
    # Calculate absolute difference between counts
    count_diff = np.abs(count_tutu - count_lc10a)
    
    # Classification with wiggle room
    tutu_dominant = matches_df[
        (count_tutu > count_lc10a) & 
        (count_diff > wiggle_room)
    ]['tutu_pre_id'].unique().tolist()
    
    lc10a_dominant = matches_df[
        (count_lc10a > count_tutu) & 
        (count_diff > wiggle_room)
    ]['tutu_post_id'].unique().tolist()
    
    balanced = matches_df[
        count_diff <= wiggle_room
    ][['tutu_pre_id', 'tutu_post_id']].apply(tuple, axis=1).tolist()
    
    return {
        'tutu_dominant': tutu_dominant,
        'lc10a_dominant': lc10a_dominant,
        'balanced': balanced
    }


#% ct_

# Set output dir
rootdir = '/Volumes/Juliana/connectome'
figdir = os.path.join(rootdir, 'analyses', 'TuTuA_LC10a_recurrent')
if not os.path.exists(figdir):
    os.makedirs(figdir)

#%%
# data
data_folder = '../../../data/flywire'
flywire_datafiles = glob.glob(os.path.join(data_folder, '*.csv.gz'))
print(flywire_datafiles)
#os.listdir(data_folder

figid = data_folder

#%%
skip_connections = True

for f in flywire_datafiles:
    parentdir, fname = os.path.split(f)
    file_name = fname.split('.')[0]
    print(file_name)
    if 'classification' in file_name:
       continue
    if skip_connections and 'connections' in file_name:
        continue
    if 'synapse_coordinates' in file_name:
        continue

    if file_name == 'fafb_v783_princeton_synapse_table':
        continue
        #file_name = 'synapses'
    command = file_name+"= pd.read_csv('"+ f +"')"
    exec(command)
    print(command)

# target connections data
target_connections = 'connections_princeton_no_threshold'
connections = pd.read_csv(os.path.join(data_folder, target_connections + '.csv.gz'))

#%%
synapses_outpath = '../../../data/flywire/fafb_synapses.parquet'
if os.path.exists(synapses_outpath):
    synapses = pd.read_parquet(synapses_outpath)
    print("Loaded synapses from pickle")
else:
    print("Creating synapses pickle")
    synapse_fpath = '../../../data/flywire/fafb_v783_princeton_synapse_table.csv.gz'
    synapses = pd.read_csv(os.path.join(synapse_fpath))

    # Rename synapse columns
    #synapses.rename(columns={'pre_root_id_720575940': 'pre_root_id', 
            #               'post_root_id_720575940': 'post_root_id'}, inplace=True)

    synapses['pre_root_id'] = [int('720575940' + str(i)) 
                                for i in synapses['pre_root_id_720575940'].values]
    synapses['post_root_id'] = [int('720575940' + str(i)) 
                                for i in synapses['post_root_id_720575940'].values]
    # Save
    synapses.to_pickle(synapses_outpath)

    #%
    synapses.to_parquet(synapses_outpath.replace('.pkl', '.parquet'))

# %%
# GET TuTuA neurons
# ========================================================
# 720575940640425294 TuTuAb (TuTuA_1?)
# 720575940622538520 TuTuAa (TuTuA_2?)

# 720575940628586261  TuTuTAa_L
# 720575940612218547 TuTuAb_L
# 720575940622538520 TuTuAa_R
# 720575940640425294  TuTuAb_R

tutu_names= {720575940628586261: 'TuTuTAa_L',
             720575940612218547: 'TuTuAb_L',
             720575940622538520: 'TuTuAa_R',
             720575940640425294: 'TuTuAb_R'}

tutu_right = [720575940622538520, 20575940640425294]
tutu_left = [720575940628586261, 720575940612218547]
tutu_types = ['TuTuAa', 'TuTuAb']
tutu_all = consolidated_cell_types[consolidated_cell_types['primary_type'].isin(tutu_types)]

# %%
# GET LC10a neurons
# =======================================================
LC10a_side = 'right'
lc10a_neurons = visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
                  & (visual_neuron_types['side']==LC10a_side)]
print(len(lc10a_neurons))

all_lc10a_neurons = visual_neuron_types[visual_neuron_types['type']=='LC10a']

#%% aoi
# GET AOTU neurons
# =======================================================
aotu_types = ['AOTU019', 'AOTU025']

aotu_names = {'AOTU025': 720575940616012061,
              'AOTU019': 720575940631517251}

aotu_id_to_name = {720575940616012061: 'AOTU025_R',
              720575940631517251: 'AOTU019_R',
              720575940633556644: 'AOTU019_L',
              720575940639182424: 'AOTU025_L'}

aotu_neurons = consolidated_cell_types[(consolidated_cell_types['primary_type'].isin(aotu_types))]
print(len(aotu_neurons))

# %%
# find connections from one set of roots to another
def find_connections(pre_root_ids, post_root_ids):
    connections_from_pre = connections[connections['pre_root_id'].isin(pre_root_ids)].copy().reset_index(drop=True)
    connections_from_pre = connections_from_pre[connections_from_pre['post_root_id'].isin(post_root_ids)]
    # merge connections from same pre and post neurons
    connections_from_pre = connections_from_pre.groupby(['pre_root_id', 'post_root_id']).agg({'syn_count': 'sum'}).reset_index()
    return connections_from_pre

def find_connection_strengths(pre_root_ids, post_root_ids):
    connections_from_pre = connections[connections['pre_root_id'].isin(pre_root_ids)].copy().reset_index(drop=True)
    connections_from_pre = connections_from_pre[connections_from_pre['post_root_id'].isin(post_root_ids)]
    # merge connections from same pre and post neurons
    connections_from_pre = connections_from_pre.groupby(['pre_root_id', 'post_root_id']).agg({'syn_strength': 'sum'}).reset_index()
    return connections_from_pre

# %%
# TuTu to LC10a
# ========================================================
min_syn_count = 5
tutu_lc10a_conn = find_connections(tutu_all['root_id'].unique(), 
                                   lc10a_neurons['root_id'].unique())
tutu_lc10a_conn_filt = tutu_lc10a_conn[tutu_lc10a_conn['syn_count']>min_syn_count]
tutu_lc10a_conn_filt.loc[tutu_lc10a_conn_filt['pre_root_id'].isin(tutu_right), 'side'] = 'right'
tutu_lc10a_conn_filt.loc[tutu_lc10a_conn_filt['pre_root_id'].isin(tutu_left), 'side'] = 'left'
print("TuTu->LC10a: ", len(tutu_lc10a_conn_filt))

# Get TuTu->LC10a synapses
tutu_lc10a_syn = synapses[synapses['pre_root_id'].isin(tutu_all['root_id'].unique()) & 
                          synapses['post_root_id'].isin(lc10a_neurons['root_id'].unique())].copy()
tutu_lc10a_syn.reset_index(drop=True, inplace=True)
print(len(tutu_lc10a_syn))

# Add TuTu name to tutu_lc10a_syn as pre_root_name, also add L or R as pre_root_side column
tutu_lc10a_syn['pre_root_name'] = tutu_lc10a_syn['pre_root_id'].map(tutu_names)
tutu_lc10a_syn['pre_root_side'] = tutu_lc10a_syn['pre_root_id'].map(lambda x: 'L' if x in tutu_left else 'R')
tutu_lc10a_syn['post_root_side'] = 'R' if LC10a_side=='right' else 'L'

#%%k
# LC10a to AOTU
# =======================================================
lc10a_aotu_conn = find_connections(lc10a_neurons['root_id'].unique(), 
                                    aotu_neurons['root_id'].unique())
lc10a_aotu_conn_filt = lc10a_aotu_conn[lc10a_aotu_conn['syn_count']>min_syn_count]
print("LC10a->AOTU: ", len(lc10a_aotu_conn_filt))

# Get LC10a->AOTU synapses (aotu_post is the same as this)
lc10a_aotu_syn = synapses[synapses['pre_root_id'].isin(lc10a_neurons['root_id'].unique()) & 
                          synapses['post_root_id'].isin(aotu_neurons['root_id'].unique())].copy()
print(len(lc10a_aotu_syn))

#%%
# Add aotu_name, aotu_type, and aotu_side to lc10a_aotu_syn
# =========================================================
# LC10a_R only connect to AOTU019_R and AOTU025_R

for aotu_id, aotu_name in aotu_id_to_name.items(): 
    lc10a_aotu_syn.loc[lc10a_aotu_syn['post_root_id'] == aotu_id, 'aotu_name'] = aotu_name
    lc10a_aotu_syn.loc[lc10a_aotu_syn['post_root_id'] == aotu_id, 'aotu_type'] = aotu_name.split('_')[0]
    lc10a_aotu_syn.loc[lc10a_aotu_syn['post_root_id'] == aotu_id, 'aotu_side'] = aotu_name.split('_')[1]


#%%
# Get LC10a back to TuTuA
# =======================================================
lc10a_tutu_conn = find_connections(lc10a_neurons['root_id'].unique(), 
                                    tutu_all['root_id'].unique())
lc10a_tutu_conn_filt = lc10a_tutu_conn[lc10a_tutu_conn['syn_count']>min_syn_count]
print("LC10a->TuTu: ", len(lc10a_tutu_conn_filt))

# Get LC10a->TuTu synapses
lc10a_tutu_syn = synapses[synapses['pre_root_id'].isin(lc10a_neurons['root_id'].unique()) & 
                          synapses['post_root_id'].isin(tutu_all['root_id'].unique())].copy()
lc10a_tutu_syn.reset_index(drop=True, inplace=True)
print(len(lc10a_tutu_syn))

# Add TuTu name to lc10a_tutu_syn as post_root_name, also add L or R as post_root_side column
lc10a_tutu_syn['post_root_name'] = lc10a_tutu_syn['post_root_id'].map(tutu_names)
lc10a_tutu_syn['post_root_side'] = lc10a_tutu_syn['post_root_id'].map(lambda x: 'L' if x in tutu_left else 'R')

#%%

# Add synapse count by grouping by pre- and post-ids and add it as a column 
tutu_lc10a_syn['syn_count'] = tutu_lc10a_syn.groupby(['pre_root_id', 'post_root_id'])['post_root_id'].transform('count')
lc10a_aotu_syn['syn_count'] = lc10a_aotu_syn.groupby(['pre_root_id', 'post_root_id'])['pre_root_id'].transform('count')
lc10a_tutu_syn['syn_count'] = lc10a_tutu_syn.groupby(['pre_root_id', 'post_root_id'])['pre_root_id'].transform('count')

#%%

# Create LC10a retinotopic color map (RIGHT side)
lc10a_post = synapses[synapses['post_root_id'].isin(lc10a_neurons['root_id'].unique())]

# In FlyWire FAFB, 'R' is actually fly's LEFT, and vice versa.
# X is left/right (inverted: decreasing numbers is rightward?)  (leftward rel to fly))
# Y is up/down in 2D (inverted: decreasing is more dorsal)
# Z is front->back (increasing is more posterior)
hue_sortby = 'pre_y'
hue_ascending = True #Fase
hue_palette = 'viridis_r'

# Sort LC10a_post by post_y
lc10a_post_sorted_ids = lc10a_post.sort_values(
                            by=hue_sortby, 
                            ascending=hue_ascending)['post_root_id'].unique()

# Create dictionary of colors
lc10a_colors = sns.color_palette(hue_palette, n_colors=len(lc10a_post_sorted_ids))
lc10a_cdict = dict(zip(lc10a_post_sorted_ids, lc10a_colors))

#%%
# AOTU color map
aotu_colors_all = sns.color_palette('cubehelix', n_colors=4) #len(aotu_neurons['root_id'].unique())).as_hex()
sns.palplot(aotu_colors_all)
#aotu_all_cdict = dict(zip(aotu_neurons['root_id'].unique(), aotu_colors_all))

aotu_cdict = {'AOTU019': aotu_colors_all[0], #aotu_all_cdict[aotu_names['AOTU019']],
              'AOTU025': aotu_colors_all[2]} #aotu_all_cdict[aotu_names['AOTU025']]}

#%%
# Plot TuTu->LC10a as scatter plot, color by side
# =======================================================
# Use marker style to indicate side
fig, axn = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)
for ai, side in enumerate(['L', 'R']):
    ax=axn[0, ai]
    sns.scatterplot(data=tutu_lc10a_syn[tutu_lc10a_syn['pre_root_side']==side], ax=ax,
                    x='post_z', y='post_y', 
                    hue='post_root_id', alpha=0.5, 
                    marker='x', s=20,
                    palette=lc10a_cdict, legend=0)
    ax.set_title(f'inputs from {side} TuTu to LC10a {LC10a_side}', 
                 fontsize=8, loc='left')
    ax.set_aspect('equal')
    ax=axn[1, ai]
    sns.scatterplot(data=lc10a_tutu_syn[lc10a_tutu_syn['post_root_side']==side], ax=ax,
                    x='pre_z', y='pre_y', 
                    hue='pre_root_id', alpha=0.5, 
                    marker='x', s=20,
                    palette=lc10a_cdict, legend=0)
    ax.set_title(f'outputs to {side} TuTu from LC10a {LC10a_side}', 
                 fontsize=8, loc='left')
    ax.set_aspect('equal')

ax.invert_yaxis()
ax.invert_xaxis()

putil.label_figure(fig, figid)

# Save fig
figname = 'connections_split_by_side'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)

#%%
# Plot each synapse distribution
alpha = 0.5
markersize = 20

# Create ScalarMappable for colorbar
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

# Create a colormap from the colors
cmap = ListedColormap(lc10a_colors)
# Create normalized values for the colorbar
norm = plt.Normalize(vmin=0, vmax=len(lc10a_post_sorted_ids)-1)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig, axn = plt.subplots(2, 3, figsize=(10, 5)
                        , sharex=True, sharey=True)
ax=axn[0, 0]
ax.set_title('LC10a->AOTU', fontsize=8, loc='left')
sns.scatterplot(data=lc10a_aotu_syn, ax=ax,
                x='pre_z', y='pre_y',
                hue='pre_root_id', 
                marker='x', s=markersize,
                palette=lc10a_cdict, legend=0,
                alpha=alpha)
# Add colorbar for retino position next to subplot [0, 0]
cbar_ax = fig.add_axes([0.33, 0.55, 0.01, 0.3])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Retinotopic Position', rotation=270, labelpad=20)


ax=axn[0, 1]
sns.scatterplot(data=lc10a_aotu_syn, ax=ax,
                x='pre_z', y='pre_y',
                hue='aotu_type', palette=aotu_cdict, legend=1,
                alpha=alpha, marker='x', s=markersize)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), frameon=False)
axn[0, 2].axis('off')

# Bottom row plots with shared colorbar
# Get global range across all syn_count data
all_syn_counts = pd.concat([tutu_lc10a_syn['syn_count'], lc10a_aotu_syn['syn_count'], lc10a_tutu_syn['syn_count']])
global_syn_range = (all_syn_counts.min(), all_syn_counts.max())

# Create shared ScalarMappable object
sm_shared = ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=global_syn_range[0], vmax=global_syn_range[1]))
sm_shared.set_array([])

ax=axn[1, 0]
synapse_side = 'post'
ax.set_title(f'TuTu->LC10a {synapse_side}', fontsize=8, loc='left')
scatter1 = sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                x=f'{synapse_side}_z', y=f'{synapse_side}_y',
                hue='syn_count', 
                palette='magma', legend=0,
                vmin=global_syn_range[0], vmax=global_syn_range[1],
                alpha=alpha, s=markersize, marker='x')

ax=axn[1, 1]
synapse_sie = 'pre'
ax.set_title(f'LC10a->AOTU {synapse_side}', fontsize=8, loc='left')
scatter2 = sns.scatterplot(data=lc10a_aotu_syn, ax=ax,
                x='pre_z', y='pre_y',
                hue='syn_count', palette='magma', legend=0,
                vmin=global_syn_range[0], vmax=global_syn_range[1],
                alpha=alpha, s=markersize, marker='x')

ax=axn[1, 2]
synapse_side = 'pre'
ax.set_title(f'LC10a->TuTu {synapse_side}', fontsize=8, loc='left')
scatter3 = sns.scatterplot(data=lc10a_tutu_syn, ax=ax,
                x=f'{synapse_side}_z', y=f'{synapse_side}_y',
                hue='syn_count', palette='magma', legend=0,
                vmin=global_syn_range[0], vmax=global_syn_range[1],
                alpha=alpha, s=markersize, marker='x')

# Add shared colorbar for syn_count
cbar_ax = fig.add_axes([0.9, 0.13, 0.01, 0.3])
cbar = fig.colorbar(sm_shared, cax=cbar_ax)
cbar.set_label('Synapse Count', rotation=270, labelpad=15)

ax.invert_yaxis()
ax.invert_xaxis()

for ax in axn.flat:
    ax.set_aspect('equal')

putil.label_figure(fig, figid)

# Save fig
figname = 'synapse_distribution'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)


#%%

# PCA on synapses (TuTu->LC10a)
# =======================================================
# Convert coords
lc10a_pca_scores = do_pca_on_synapses(tutu_lc10a_syn, xvar='post_z', yvar='post_y')

#% # Add PCA scores to the original dataframe
tutu_lc10a_syn_pca = pd.concat([tutu_lc10a_syn, lc10a_pca_scores], axis=1)

fig = plot_pca_transformed(tutu_lc10a_syn, tutu_lc10a_syn_pca, lc10a_cdict)
putil.label_figure(fig, figid)

# Save fig
figname = 'check_pca_tutu-lc10a'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)

#%%
tutu_lc10a_syn_pca_binned = bin_pca_scores(tutu_lc10a_syn_pca)
#%
bin_cmap = 'viridis_r'
fig = plot_joint_pca_scores(tutu_lc10a_syn_pca_binned, tutu_lc10a_syn_pca, 
                            lc10a_cdict, bin_cmap=bin_cmap,
                            markersize=10, marker='x',
                            marginal_marker='o', marginal_markersize=20)
putil.label_figure(fig, figid)

# Save fig
figname = 'joint_pca_scores_tutu-lc10a'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)

#%% Do the same for LC10a->TuTu
# =======================================================
lc10a_pca_scores = do_pca_on_synapses(lc10a_tutu_syn, xvar='pre_z', yvar='pre_y')
lc10a_tutu_syn_pca = pd.concat([lc10a_tutu_syn, lc10a_pca_scores], axis=1)
fig = plot_pca_transformed(lc10a_tutu_syn, lc10a_tutu_syn_pca, lc10a_cdict,
                           xvar='pre_z', yvar='pre_y', hue_var='pre_root_id')
putil.label_figure(fig, figid)
figname = 'check_pca_lc10a-tutu'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)

#%%
lc10a_tutu_syn_pca_binned = bin_pca_scores(lc10a_tutu_syn_pca)
#%
bin_cmap = 'viridis_r'
fig = plot_joint_pca_scores(lc10a_tutu_syn_pca_binned, lc10a_tutu_syn_pca, 
                            lc10a_cdict, bin_cmap=bin_cmap, 
                            hue_var='pre_root_id',
                            markersize=10, marker='x',
                            marginal_marker='o', marginal_markersize=20)
putil.label_figure(fig, figid)


#%%
matches_df = synapse_matches(tutu_lc10a_syn, lc10a_tutu_syn)

#%% 
# Visualize the distance distribution
fig1 = hist_synapse_distances(matches_df)
fig2 = scatter_synapse_counts(matches_df, lc10a_cdict)
    
# Define threshold for "likely same" vs "likely different"
distance_threshold = matches_df['distance'].quantile(0.5)  # Median as threshold

likely_same = matches_df[matches_df['distance'] <= distance_threshold]
likely_different = matches_df[matches_df['distance'] > distance_threshold]

print(f"\nClassification (threshold: {distance_threshold:.1f}μm):")
print(f"  Likely same connections: {len(likely_same)} ({len(likely_same)/len(matches_df)*100:.1f}%)")
print(f"  Likely different connections: {len(likely_different)} ({len(likely_different)/len(matches_df)*100:.1f}%)")

putil.label_figure(fig2, figid)

# Save fig
figname = 'synapse_count_correlation'
putil.save_fig(figname, fig, figid, figdir, save_svg=True)
print(figname)

#%%



   
    #%% 
# Extract directional neurons using normalized counts
directional_neurons = extract_directional_neurons(matches_df, use_normalized=True)

print(f"\nDirectional Analysis:")
print(f"  TuTu dominant (TuTu->LC10a > LC10a->TuTu): {len(directional_neurons['tutu_dominant'])} neurons")
print(f"  LC10a dominant (LC10a->TuTu > TuTu->LC10a): {len(directional_neurons['lc10a_dominant'])} neurons")
print(f"  Balanced (TuTu->LC10a = LC10a->TuTu): {len(directional_neurons['balanced'])} pairs")

# Create color-coded scatter plot using seaborn
fig3, ax = plt.subplots(1, 1, figsize=(6, 5))

# Create a new column for neuron type classification
matches_df['neuron_type'] = 'Balanced'

# Classify each connection into 3 main categories with wiggle room
# Use normalized counts if available
if 'syn_count_tutu_norm' in matches_df.columns:
    count_tutu = matches_df['syn_count_tutu_norm']
    count_lc10a = matches_df['syn_count_lc10a_norm']
    print("Using normalized synapse counts for classification")
else:
    count_tutu = matches_df['syn_count_tutu']
    count_lc10a = matches_df['syn_count_lc10a']
    print("Using raw synapse counts for classification")

max_count = max(count_tutu.max(), count_lc10a.max())
wiggle_room = 0.1 * max_count  # 10% wiggle room

for idx, row in matches_df.iterrows():
    syn_tutu = count_tutu.iloc[idx]
    syn_lc10a = count_lc10a.iloc[idx]
    count_diff = abs(syn_tutu - syn_lc10a)
    
    # Classification with wiggle room
    if syn_tutu > syn_lc10a and count_diff > wiggle_room:
        matches_df.loc[idx, 'neuron_type'] = 'TuTu Dominant'
    elif syn_lc10a > syn_tutu and count_diff > wiggle_room:
        matches_df.loc[idx, 'neuron_type'] = 'LC10a Dominant'
    else:  # count_diff <= wiggle_room
        matches_df.loc[idx, 'neuron_type'] = 'Balanced'

# Define colors for each type
type_colors = {
    'Balanced': 'lightgray',
    'TuTu Dominant': 'red',
    'LC10a Dominant': 'blue',
    'Neither': 'black'
}

# Plot to check - use normalized counts if available
if 'syn_count_tutu_norm' in matches_df.columns:
    sns.scatterplot(data=matches_df, x='syn_count_tutu_norm', y='syn_count_lc10a_norm', 
                    hue='neuron_type', palette=type_colors, alpha=0.7, s=50, ax=ax)
    ax.set_xlabel('TuTu→LC10a Synapse Fraction')
    ax.set_ylabel('LC10a→TuTu Synapse Fraction')
    ax.set_title('Directional Neuron Classification (Normalized)')
else:
    sns.scatterplot(data=matches_df, x='syn_count_tutu', y='syn_count_lc10a', 
                    hue='neuron_type', palette=type_colors, alpha=0.7, s=50, ax=ax)
    ax.set_xlabel('TuTu→LC10a Synapse Count')
    ax.set_ylabel('LC10a→TuTu Synapse Count')
    ax.set_title('Directional Neuron Classification')

# Add diagonal line for perfect correlation
if 'syn_count_tutu_norm' in matches_df.columns:
    max_count = max(count_tutu.max(), count_lc10a.max())
else:
    max_count = max(matches_df['syn_count_tutu'].max(), matches_df['syn_count_lc10a'].max())
ax.plot([0, max_count], [0, max_count], 'k--', alpha=0.5, label='Perfect correlation')
ax.set_aspect('equal')

# Move legend outside the plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the directional analysis plot
putil.label_figure(fig3, figid)
figname = 'directional_neuron_classification'
putil.save_fig(figname, fig3, figid, figdir, save_svg=True)
print(figname)

# Print some examples
if len(directional_neurons['tutu_dominant']) > 0:
    print(f"  TuTu dominant examples: {directional_neurons['tutu_dominant'][:5]}")
if len(directional_neurons['lc10a_dominant']) > 0:
    print(f"  LC10a dominant examples: {directional_neurons['lc10a_dominant'][:5]}")

#

#%%
# Assign neuron type to tutu_lc10a_syn using matches_df['tutu_post_id'] to match with tutu_lc10a_syn['post_root_id']
tutu_lc10a_syn['neuron_type'] = 'Neither'
for idx, row in matches_df.iterrows():
    tutu_lc10a_syn.loc[tutu_lc10a_syn['post_root_id'] == row['tutu_post_id'], 'neuron_type'] = row['neuron_type']

# Plot z, y position and color by neuron type
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(data=tutu_lc10a_syn, x='pre_z', y='pre_y', ax=ax,
                hue='neuron_type', palette=type_colors, 
                alpha=0.7, s=20, marker='x') 
ax.set_aspect('equal')
ax.set_title("LC10a post that receives more TuTu is red")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
ax.invert_yaxis()
ax.invert_xaxis()

putil.label_figure(fig, figid)

#%%
# Do the same but assign neuron_type for lc10a_tutu_syn using matches_df['tutu_post_id'] to match with lc10a_tutu_syn['pre_root_id']
lc10a_tutu_syn['neuron_type'] = 'Neither'
for idx, row in matches_df.iterrows():
    lc10a_tutu_syn.loc[lc10a_tutu_syn['pre_root_id'] == row['tutu_post_id'], 'neuron_type'] = row['neuron_type']
    
# Plot z, y position and color by neuron type
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(data=lc10a_tutu_syn, x='post_z', y='post_y', ax=ax,
                hue='neuron_type', palette=type_colors, 
                alpha=0.7, s=20, marker='x') 
ax.set_aspect('equal')
ax.set_title("LC10a pre that receives more TuTu is red")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
ax.invert_yaxis()
ax.invert_xaxis()

putil.label_figure(fig, figid)
figname = 'directional_neuron_classification'






#%%

# Merge tutu_lc10a_conn_filt and lc10a_tutu_conn_filt in a new dataframe
# match post_root_id in tutu_lc10a_conn_fitl to pre_root_id in lc10a_tutu_conn_filt
# first make a copy of tutu_lc10a_conn_filt and rename 'post_root_id' to 'lc10_id' and pre_root_id to 'tutu_id'
tutu_lc10a_conn_filt_copy = tutu_lc10a_conn_filt.copy()
tutu_lc10a_conn_filt_copy.rename(columns={'post_root_id': 'lc10_id', 'pre_root_id': 'tutu_id'}, inplace=True)
# Now make a copy of lc10a_tutu_conn_filt and rename 'post_root_id' to 'tutu_id' and pre_root_id to 'lc10_id'
lc10a_tutu_conn_filt_copy = lc10a_tutu_conn_filt.copy()
lc10a_tutu_conn_filt_copy.rename(columns={'post_root_id': 'tutu_id', 'pre_root_id': 'lc10_id'}, inplace=True)

# Now merge the two dataframes, and match pairs of lc10_id and tutu_id
tutu_lc10a = pd.concat([tutu_lc10a_conn_filt_copy[['tutu_id', 'lc10_id', 'syn_count']], 
                        lc10a_tutu_conn_filt_copy[['tutu_id', 'lc10_id', 'syn_count']]], axis=0)
tutu_lc10a







   
    
# %%
import zipfile
# Plot LC10a skeletons
# =======================================================
# Read a zip file of LC10a skeletons
skel_root = os.path.join(rootdir, 'data', 'FlyWire')
all_skel = zipfile.ZipFile(os.path.join(skel_root, 'sk_lod1_783_healed.zip'))
all_skel_files = all_skel.namelist()

# Read each skeleton file
for lc10a_id in lc10a_neurons['root_id'].unique():
    skel_file = all_skel_files[lc10a_id]
    skeleton = all_skel.read(skel_file)
    print(skeleton)

#%%
from fafbseg import flywire
