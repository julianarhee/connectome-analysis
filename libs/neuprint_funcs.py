#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:07:10 2024

@File: neuprint_funcs.py
@Time: 2024/05/09 11:07:10
@Author: julianarhee

"""

import os
import numpy as np
import pandas as pd
import pylab as pl
import pickle as pkl
import re

import neuprint as neu
from neuprint import NeuronCriteria as NC

from sklearn import mixture

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform


from neuprint import Client
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#%%
def get_neuprint_client(dataset='male-cns:v0.9'):
    # Load token from shell (IDE doesn't inherit .zshrc env vars)
    import subprocess
    try:
        token = subprocess.check_output(
            ['zsh', '-c', 'source ~/.zshrc && echo $NEUPRINT_APPLICATION_CREDENTIALS'],
            text=True
        ).strip()
        print(f"✓ Token loaded from shell (length: {len(token)})")
    except Exception as e:
        print(f"✗ Failed to load token: {e}")
        token = None
    #%
    c = Client('neuprint.janelia.org', dataset=dataset, token=token)
    c.fetch_version()
    return c

#%%
# =============== UTILITY FUNCTIONS ===============a
def merge_properties_and_group(neuron_df, conn_df, 
                               groupby_cols=['type_pre', 'type_post', 'roi_noside']):
    '''
    Merge neuron properties and group by type and ROI.
    Args:
        neuron_df: DataFrame, the neuron dataframe
        conn_df: DataFrame, the connection dataframe
    Returns:
        conn_df: DataFrame, the connection dataframe with the merged properties and grouped by type and ROI
    '''
    conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
    conn_df = extract_side_from_conn_df(conn_df)
    #%
    # Group across side
    conn_df = conn_df.groupby(groupby_cols, as_index=False)['weight'].sum()

    return conn_df

def extract_side_from_conn_df(conn_df):
    '''
    Extract the side from the instance column and the ROI without the side from the roi column.
    Args:
        conn_df: DataFrame, the connection dataframe
    Returns:
        conn_df: DataFrame, the connection dataframe with the side and ROI without the side columns
    '''
    # Handle None/NaN values in instance columns
    conn_df['side_pre'] = conn_df['instance_pre'].apply(lambda x: x.split('_')[-1] if pd.notna(x) and x is not None else None)
    conn_df['side_post'] = conn_df['instance_post'].apply(lambda x: x.split('_')[-1] if pd.notna(x) and x is not None else None)
    conn_df['roi_noside'] = conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
    return conn_df

def fetch_neuron_types(client, neuron_types):
    """Fetch neurons for one or more cell types and return combined dataframe."""
    if isinstance(neuron_types, str):
        neuron_types = [neuron_types]
    
    dfs, roi_dfs = [], []
    for ntype in neuron_types:
        df, roi_df = neu.fetch_neurons(NC(type=ntype, client=client))
        dfs.append(df)
        roi_dfs.append(roi_df)
        print(f"  {ntype}: {len(df)} neurons")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, pd.concat(roi_dfs, ignore_index=True), combined_df['bodyId'].unique()


def split_ids_by_side(df):
    """Split neuron bodyIds by L/R hemisphere."""
    return (df[df['somaSide'] == 'L']['bodyId'].unique(),
            df[df['somaSide'] == 'R']['bodyId'].unique())


def split_ids_by_side_from_matrix(conn_matrix, src_df, target_df):
    """Split neuron bodyIds by L/R hemisphere using the sorted matrix order."""
    # Get all source and target IDs in matrix order
    src_ids = conn_matrix.index.tolist()
    tgt_ids = conn_matrix.columns.tolist()
    
    # Create mappings for quick lookup
    src_side_map = dict(zip(src_df['bodyId'], src_df['somaSide']))
    tgt_side_map = dict(zip(target_df['bodyId'], target_df['somaSide']))
    
    # Split source IDs by side
    src_L = [id for id in src_ids if src_side_map.get(id) == 'L']
    src_R = [id for id in src_ids if src_side_map.get(id) == 'R']
    
    # Split target IDs by side  
    tgt_L = [id for id in tgt_ids if tgt_side_map.get(id) == 'L']
    tgt_R = [id for id in tgt_ids if tgt_side_map.get(id) == 'R']
    
    return src_L, src_R, tgt_L, tgt_R

# sort index
def sort_index(conn_matrix, conn_df=None, axis='rows', 
               sort_by=None, sorted_var_name=None,
               sort_weights=True, weight_var='weight',
               sort_by_lut=None):
    '''
    Sort rows or columns of a connection matrix by weight or ROI.
    Args:
        conn_matrix: DataFrame, the connection matrix to sort
        conn_df: DataFrame, the connection dataframe
        axis: str, 'rows' or 'cols', the axis to sort
        sort_by: str, 'weight' or 'roi', the column to sort by (or None to not sort)
        sorted_var_name: str, the name of the index or columns to sort, e.g., 'type_pre' (if None, 
                     assumes conn_matrix is a hierarchical index with .index.name and .columns.name)
        sort_weights: bool, whether to sort by weights as well (default True)
        weight_var: str, the column to use for the weights
        sort_by_lut: dict, a dictionary of the values to sort by and their order to use
                    e.g., {'roi1': 0, 'roi2': 1, 'roi3': 2} for CUSTOM sorting
    Returns:
        sorted_ix: list, the sorted indices
    '''
    if axis == 'rows':
        summed_axis = 1 # Sum across columns, get total weight for each row
        sorted_var_name = conn_matrix.index.name if sorted_var_name is None else sorted_var_name
    else:
        summed_axis = 0 # Sum across rows, get total weight for each column 
        sorted_var_name = conn_matrix.columns.name if sorted_var_name is None else sorted_var_name
        
    if isinstance(sort_by, str) and sort_by == 'weight':
        sorted_ix = conn_matrix.sum(axis=summed_axis).sort_values(ascending=False).index.tolist()
    else: #'roi':
        assert conn_df is not None, "conn_df is required for sorting by ROI"
        assert sort_by in conn_df.columns, f"Column {sort_by} not found in conn_df"
        #sorted_ix = conn_df.sort_values(by=sort_by).index.tolist()
        # For each index, get its sort_by value
        conn_df_info = conn_df.groupby(sorted_var_name, as_index=False)\
                              .apply(lambda x: x[sort_by].unique()[0])\
                              .rename(columns={None: sort_by})

        if sort_weights:
            assert weight_var in conn_df.columns, f"Column {weight_var} not found in conn_df"
            conn_df_weights = conn_df.groupby(sorted_var_name, as_index=False)[weight_var].sum()#\
                            #.sort_values(by=weight_var, ascending=False) 
            # merge conn_df_info and conn_df_weights
            conn_df_info = conn_df_info.merge(conn_df_weights, on=sorted_var_name, how='left')
                              
        if sort_by_lut is not None:
            assert isinstance(sort_by_lut, dict), "sort_by_lut must be a dictionary"
            conn_df_info[f'{sort_by}_ix'] = [sort_by_lut[v] for v in conn_df_info[sort_by]]
            if sort_weights:
                sort_by_vars_list = [f'{sort_by}_ix', weight_var]
            else:
                sort_by_vars_list = [f'{sort_by}_ix']
                
        else:
            if sort_weights:
                sort_by_vars_list = [sort_by, weight_var]
            else:
                sort_by_vars_list = [sort_by]
                                
        sorted_ix = conn_df_info.sort_values(by=sort_by_vars_list,
                                                 ascending=True)[sorted_var_name].tolist()
#         else: 
#             if sort_weights:
#                 # Sort by both
#                 sorted_ix = conn_df_info.sort_values(by=[sort_by, weight_var],\
#                                             ascending=False)[sorted_var_name].tolist()
#             else:
#                 sorted_ix = conn_df_info.sort_values(by=sort_by)[sorted_var_name].tolist()
     #else:
    #raise ValueError(f"Invalid sorting axis: {sort_by}")

    return sorted_ix

def sort_matrix_labels(conn_matrix, conn_df=None, 
                       sort_rows_by=None, sort_cols_by=None,
                       sorted_var_name=None,
                       sort_weights=True, weight_var='weight',
                       sort_by_lut_row=None, sort_by_lut_col=None):
    '''
    Sort the rows and columns of a connection matrix.
    Args:
        conn_matrix: DataFrame, the connection matrix to sort
        conn_df: DataFrame, the connection dataframe
        sort_rows_by: str, the column to sort rows by, or None to not sort rows
        sort_cols_by: str, the column to sort columns by, or None to not sort columns
        sort_weights: bool, whether to sort by weights as well
        weight_var: str, the column to use for the weights
        sort_by_lut_row: dict, a dictionary of the values to sort rows by and their order to use
                    e.g., {'roi1': 0, 'roi2': 1, 'roi3': 2} for CUSTOM sorting
        sort_by_lut_col: dict, a dictionary of the values to sort columns by and their order to use
                    e.g., {'roi1': 0, 'roi2': 1, 'roi3': 2} for CUSTOM sorting
    Returns:
        conn_matrix: DataFrame, the sorted connection matrix
    '''
    sorted_rows = None
    sorted_cols = None
    if sort_rows_by is not None:
        if isinstance(sort_rows_by, (list, np.ndarray)):
            sorted_rows = sort_rows_by
        else: # sort by string (column in conn_df)
            sorted_rows = sort_index(conn_matrix, conn_df=conn_df, 
                                     axis='rows', sort_by=sort_rows_by, 
                                     sorted_var_name=sorted_var_name,
                                     sort_weights=sort_weights, 
                                     weight_var=weight_var,
                                     sort_by_lut=sort_by_lut_row)
        assert len(sorted_rows) == conn_matrix.shape[0], "Number of sorted rows does not match number of rows in conn_matrix"
    if sort_cols_by is not None:
        if isinstance(sort_cols_by, (list, np.ndarray)):
            sorted_cols = sort_cols_by
        else: # sort by string (column in conn_df)  
            sorted_cols = sort_index(conn_matrix, conn_df=conn_df, 
                                     axis='cols', sort_by=sort_cols_by,
                                     sorted_var_name=sorted_var_name,
                                     sort_weights=sort_weights, 
                                     weight_var=weight_var,
                                     sort_by_lut=sort_by_lut_col)
        assert len(sorted_cols) == conn_matrix.shape[1], "Number of sorted columns does not match number of columns in conn_matrix"
    conn_matrix = conn_matrix.reindex(index=sorted_rows, columns=sorted_cols)
    
    return conn_matrix


# ============================== NORMALIZATION FUNCTIONS ==============================
def norm_by_specified_inputs(conn_df, group_col='instance_post'):
    '''
    Normalize the target weights of a connection dataframe by the 
    total inputs in the specified dataframe. Assumes that ALL inputs to 
    a given target are included in conn_df.
    Args:
        conn_df: DataFrame, the connection dataframe
        group_col: str, the column to group by
    Returns:
        conn_df: DataFrame, the connection dataframe with normalized weights
    '''
    total_weights = conn_df.groupby(group_col, as_index=False)['weight'].sum()
    for group_val, df_ in conn_df.groupby(group_col):
        df_['percent_of_total'] = df_['weight'] / total_weights[total_weights[group_col]==group_val]['weight'].values[0]
        conn_df.loc[conn_df[group_col]==group_val, 'percent_of_total'] = df_['percent_of_total']
    return conn_df

def get_and_norm_by_total_inputs(conn_df, c, normalize_group_col='type_post',
                                 groupby_cols=['type_pre', 'type_post', 'roi_noside'],
                                 min_total_weight=10):
    '''
    Get all targets of each type of SRC neurons, then noramlize those targets by 
    their total inputs. Fetches all sources to conn_df['type_post'] types, normalizes by total inputs to target types,
    and returns the subset of connections between source and target types. 
    Use when normalizing a given neuron type's OUTPUTs.
    Args:
        conn_df: DataFrame, the connection dataframe
        normalize_group_col: str, the column to normalize by (default: 'type_post'), 
        which means that total inputs for each type_post is used to normalize the weights.
    Returns:
        normalized_output_conn_df: DataFrame, the normalized connection dataframe
    '''
    src_types = conn_df['type_pre'].unique() # source types
    target_types = conn_df['type_post'].unique()
    
    # Get all inputs to target types
    inputs_to_target_neurons, inputs_to_target_conns = neu.fetch_adjacencies(
                                          sources=None,
                                          targets=NC(type=target_types),
                                          client=c, min_total_weight=min_total_weight)
    
    # Merge columns from neuron_df to conn_df:
    inputs_to_target_conns = neu.merge_neuron_properties(inputs_to_target_neurons, 
                                        inputs_to_target_conns, ['type', 'instance'])

    # Add roi_noside if in groupby_cols:
    if 'roi_noside' in groupby_cols:
        inputs_to_target_conns = add_roi_noside(inputs_to_target_conns)

    # Group by type and/or ROI:
    groupby_type = normalize_group_col == 'type_post'         
    if groupby_type:
        inputs_to_target_conns = inputs_to_target_conns.groupby(groupby_cols, \
                                                            as_index=False)['weight'].sum()
        assert normalize_group_col == 'type_post', "normalize_group_col must be 'type_post' when groupby_type is True"
        
    # Normalize by total inputs to each target type
    inputs_to_target_conns = norm_by_specified_inputs(inputs_to_target_conns, 
                                                           group_col=normalize_group_col)
    # Select subset of connections between source and target types
    normalized_output_conn_df = inputs_to_target_conns[\
                                        inputs_to_target_conns['type_pre'].isin(src_types)].copy()
    
    return normalized_output_conn_df

def add_roi_noside(conn_df):
    '''
    conn_df (2nd output of neu.fetch_adjacencies) splits ROI by side (R/L),
    split again to get ROI without side.     
    Args:
        conn_df: DataFrame, the connection dataframe
    Returns:
        conn_df: DataFrame, the connection dataframe with roi_noside column
    '''
    conn_df['roi_noside'] = conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
    return conn_df


def get_conn_all_outputs(source_type, client=None, return_both=True,
                         return_matrix_only=False,
                         norm_by_all_other_inputs=False, weight_var='percent_of_total',
                         min_total_weight=10,
                         groupby_cols=['type_pre', 'type_post', 'roi_noside']):
    '''
    Get all outputs from a given source type. If norm_by_all_other_inputs is True,
    normalize all targets by THEIR total inputs. Otherwise, normalize by the total
    inputs to the target type from the specified sources.
    Args:
        source_type: str, the type of the source neurons
        client: neuprint.Client, the neuprint client
        return_matrix: bool, whether to return the connection matrix
        norm_by_all_other_inputs: bool, whether to normalize by the total inputs to all other inputs
        groupby_cols: list, the columns to group by when merging neuron properties and grouping
    Returns:
    '''
    if isinstance(source_type, str):
        source_nc = NC(type=f'{source_type}.*')
    else:
        assert isinstance(source_type, list), "source_type must be a string or a list of strings"
        source_nc = NC(type=source_type)
    neuron_df, conn_df = neu.fetch_adjacencies(sources=source_nc, 
                                                targets=None,
                                                client=client,
                                                min_total_weight=min_total_weight)
    
    conn_df = merge_properties_and_group(neuron_df, conn_df,
                                         groupby_cols=groupby_cols) 
    
    # Normalize all targets by THEIR total inputs
    #if weight_var != 'percent_of_total':
    if norm_by_all_other_inputs:
        conn_df = get_and_norm_by_total_inputs(conn_df, client, 
                                #groupby_type=True, 
                                normalize_group_col='type_post',
                                groupby_cols=groupby_cols, 
                                min_total_weight=min_total_weight)
    else:
        conn_df = norm_by_specified_inputs(conn_df, group_col='type_post')
            
    if return_matrix_only or return_both:
        conn_matrix = connection_table_to_matrix(conn_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
        if return_both:
            return conn_df, conn_matrix
        else:
            return conn_matrix
    else:
        return conn_df

def get_conn_all_inputs(target_type, target_list=None, client=None, return_both=True,
                        return_matrix_only=False, weight_var='percent_of_total',
                        min_total_weight=10,
                        groupby_cols=['type_pre', 'type_post', 'roi_noside']):
    '''
    Get all inputs to a given target type. If target_list is not None, get all inputs to the targets in the list.
    Args:
        target_type: str, the type of the target neurons
        target_list: list, the list of target types
        client: neuprint.Client, the neuprint client
        return_both: bool, whether to return the connection dataframe and matrix
        return_matrix_only: bool, whether to return the connection matrix only
        weight_var: str, the column to use for the weights
        min_total_weight: int, the minimum total weight for a connection to be included
        groupby_cols: list, the columns to group by when merging neuron properties and grouping
    Returns:
    if return_both:
        return conn_df, conn_matrix
    else:
        return conn_matrix
    '''
    if target_list is not None:
        targets = target_list
    else:
        if isinstance(target_type, str):
            targets = NC(type=f'{target_type}.*')
        else:
            assert isinstance(target_type, list), "target_type must be a string or a list of strings"
            targets = NC(type=target_type)
            
    neuron_df, conn_df = neu.fetch_adjacencies(sources=None,
                                    targets=targets, 
                                    client=client,
                                    min_total_weight=min_total_weight)
    conn_df = merge_properties_and_group(neuron_df, conn_df,
                                         groupby_cols=groupby_cols) 
        
    # Normalize targets (already have ALL their inputs)
    conn_df = norm_by_specified_inputs(conn_df, group_col='type_post')
    
    if return_matrix_only or return_both:
        conn_matrix = connection_table_to_matrix(conn_df,
                                    weight_col=weight_var,
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
        if return_both:
            return conn_df, conn_matrix
        else:
            return conn_matrix
    else:
        return conn_df

def get_normed_filepath(processed_dir='/Volumes/Juliana/connectome/analyses/processed_data',
                        neuron_type='LC', io_type='inout'):
    '''
    Get the filepath for the normalized data.
    Args:
        processed_dir: str, the directory to save the normalized data
        neuron_type: str, the type of the neuron
        io_type: str, the type of input or output
    Returns:
        normed_data_fpath: str, the filepath for the normalized data
    '''
    normed_data_fpath = os.path.join(processed_dir, f'{neuron_type}_normed_{io_type}.pkl')
    return normed_data_fpath

def load_normed_data(neuron_type='LC', io_type='input',
                     processed_dir='/Volumes/Juliana/connectome/analyses/processed_data',
                     normed_data_fpath=None, client=None, create_new=False): 
    '''
    Load the normalized data from the filepath. Data normalized by total inputs to each target type.
    This is currently specific to LC_normed_inputs.pkl and LC_normed_outputs.pkl files, 
    created by grouping by roi_noside, type_pre, and type_post.
    Args:
        normed_data_fpath: str, the filepath for the normalized data
        create_new: bool, whether to create new normalized data
    Returns:
        conn_df: DataFrame, the connection dataframe
        matrix: DataFrame, the connection matrix
    '''
    if normed_data_fpath is None:
        normed_data_fpath = get_normed_filepath(processed_dir=processed_dir, neuron_type=neuron_type, 
                                                io_type=io_type) 
    #else:
    assert os.path.exists(normed_data_fpath), "normed_data_fpath does not exist: {normed_data_fpath}"
    
    # Get neuron_type and io_type:
    if neuron_type is None:
        neuron_type = os.path.split(normed_data_fpath)[1].split('_')[0]
    if io_type is None:
        io_type = os.path.split(normed_data_fpath)[1].split('_')[-1]
    print("Neuron type: {neuron_type}, IO type: {io_type}") 
    
    try: 
        assert create_new is False, "create_new must be False to load existing file"
        with open(normed_data_fpath, 'rb') as f:
            normed_data_tmp = pkl.load(f)
        conn_df = normed_data_tmp['conn_df']
        matrix = normed_data_tmp['conn_matrix']
        print(f"Loaded: {normed_data_fpath}")    
        assert conn_df.groupby('type_post')['percent_of_total'].sum().max() < 1, 'target weights should not sum to 1 if percent_of_total is used'
    except Exception as e: #FileNotFoundError:
        # Specific case for LCs: get all LC types that start with LC followed by a number
        print("Creating and normalizing all data...")
        if neuron_type == 'LC':
            targets = get_all_LCs(client=client)
        else:
            targets = neuron_type 
        
        if 'outputs' in normed_data_fpath:
            conn_df, matrix = get_conn_all_outputs(targets, client=client, 
                                            return_both=True,
                                            norm_by_all_other_inputs=True, 
                                            weight_var='percent_of_total')
        else:
            conn_df, matrix = get_conn_all_inputs(targets, client=client, 
                                            return_both=True,
                                            weight_var='percent_of_total')
        normed_data_tmp = {'conn_df': conn_df, 'conn_matrix': matrix}
        with open(normed_data_fpath, 'wb') as f:
            pkl.dump(normed_data_tmp, f)
        print(f"Saved: {normed_data_fpath}")
    return conn_df, matrix


def get_all_LCs(client=None):
    '''
    Get all LC types that start with LC followed by a number.
    Args:
        client: Client object, the neuprint client
    Returns:
        all_LCs: list, the list of all LC types
    '''
    # Get all LCs
    LC_neurons_df, LC_roi_counts_df = neu.fetch_neurons(NC(type='LC.*', client=client))
    LC_neurons_df.head()
 
    # Only get LC types that start with LC followed by a number
    all_LCs_returned = LC_neurons_df['type'].unique()
    all_LCs = sorted([lc for lc in all_LCs_returned if re.match(r'^LC\d', lc)\
                        and 'unclear' not in lc], key=util.natsort)
    return all_LCs





# ============================== SPATIAL FUNCTIONS ==============================
def get_den_ax_loc(cell,num_gauss=3, plot=False, client=None,
                   color_dendrites='r', color_axons='b'):
    """
    Function determines axon and dendrite location from cell type
    Returns: 
        ax_term: axon terminal mean location
        den_term: mean locations of dendrite terminals
    How it works:
        Function takes the mean pre-synapse location to be the axon terminal
        Then makes a guassian mixture model on post synapse terminals
        Where the two don't overlap are where dendrites are
    Future extensions: 
        Use gaussian mixture modelling to get terminal location for
        cells with multiple axon terminal sites
    Limitations: 
        You should know in advance how the neurite tree looks to specify the
        number of gaussians
    
    """
    
    syndf = neu.fetch_synapses(cell, client=client)
    pre_post = syndf['type']
    # get pre and post synapse locations
    pre = pd.Series.to_numpy(pre_post=='pre')
    post = pd.Series.to_numpy(pre_post=='post')
    syn_locs = np.array([syndf['x'], syndf['y'], syndf['z']])
    
    cdx = pd.Series.to_numpy(syndf['confidence']>0.9) # Can relax this if needs be
    syn_locs = np.transpose(syn_locs)
    pre_locs = syn_locs[pre&cdx,:]
    post_locs = syn_locs[post&cdx,:]
    
    # get mean pre location: axons are the cell's pre-synaptic sites
    mn_pre = np.mean(pre_locs,axis=0)
    sd_pre = np.std(pre_locs,axis=0)
    #
    gm = mixture.GaussianMixture(n_components=num_gauss, random_state=0).fit(post_locs)
    # Gaussian mixture model with two gaussians
    mn_post = gm.means_

    #  
    ax_term = mn_pre
    # get mean locations of all post-synaptic sites, include the ones by axon term
    den_dx = np.sqrt(np.sum(np.square(mn_post-mn_pre),axis=1))
    # filter out the ones that correspond to the axon terminal
    dx = [True]*num_gauss #[True, True, True]
    ax_dx = np.argmin(den_dx)
    dx[ax_dx] = False
    den_term = mn_post[dx,:]
    
    if plot:
        fig = pl.figure()
        ax = fig.add_subplot(projection='3d')
        cdict = {'pre': color_axons, 'post': color_dendrites} # dendrites are the cell's post-synaptic sites; axons are its pre-synaptic sites
        col_list = [cdict[v] for v in syndf['type']]
        ax.scatter(syndf['x'], syndf['y'], syndf['z'], c=col_list, s=3)
        # plot ax
        ax.plot(ax_term[0], ax_term[1], ax_term[2], c=color_axons, marker='o', markersize=10)
        # plot den
        for d in den_term:
            ax.plot(d[0], d[1], d[2], c=color_dendrites, marker='o', markersize=10)
         
    return ax_term, den_term

def get_axo_den_locs_for_cell_ids(cell_ids: list, client=None):
    # For each ID, get inputs (dendrite labels) and outputs (axon labels)

    d_list = []
    for cell in cell_ids: #LC10_ids['LC10a'].dropna():
        ax_term, den_term = get_den_ax_loc(int(cell), num_gauss=2, client=client)
        arr_ = np.vstack([ax_term, den_term])
        d_ = pd.DataFrame(arr_, columns=['x', 'y', 'z'], index=range(arr_.shape[0]))
        n_axo = 1 if len(ax_term.shape)==1 else ax_term.shape[0] # returns array 
        n_den = 1 if len(den_term.shape)==1 else den_term.shape[0] # if returns an array, take 1st dim
        axo_labels = np.tile('axon', n_axo)
        den_labels = np.tile('dendrite', n_den)
        labels = flatten([axo_labels, den_labels])
        d_['type'] = labels
        d_['cell'] = [cell] * len(labels)
        d_list.append(d_)  
    prepost = pd.concat(d_list, axis=0)
    prepost = prepost.reset_index().rename(columns={'index': 'n_terminals'})
    
    return prepost

# ============================== CONNECTION MATRIX FUNCTIONS ==============================
def connection_table_to_matrix(conn_df, group_cols='bodyId', weight_col='weight', sort_by=None, make_square=False):
    """
    Given a weighted connection table, produce a weighted adjacency matrix.

    Args:
        conn_df:
            A DataFrame with columns for pre- and post- identifiers
            (e.g. bodyId, type or instance), and a column for the
            weight of the connection.

        group_cols:
            Which two columns to use as the row index and column index
            of the returned matrix, respetively.
            Or give a single string (e.g. ``"body"``, in which case the
            two column names are chosen by appending the suffixes
            ``_pre`` and ``_post`` to your string.

            If a pair of pre/post values occurs more than once in the
            connection table, all of its weights will be summed in the
            output matrix.

        weight_col:
            Which column holds the connection weight, to be aggregated for each unique pre/post pair.

        sort_by:
            How to sort the rows and columns of the result.
            Can be two strings, e.g. ``("type_pre", "type_post")``,
            or a single string, e.g. ``"type"`` in which case the suffixes are assumed.

        make_square:
            If True, insert rows and columns to ensure that the same IDs exist in the rows and columns.
            Inserted entries will have value 0.0

    Returns:
        DataFrame, shape NxM, where N is the number of unique values in
        the 'pre' group column, and M is the number of unique values in
        the 'post' group column.

    Example:

        .. code-block:: ipython

            In [1]: from neuprint import fetch_simple_connections, NeuronCriteria as NC
               ...: kc_criteria = NC(type='KC.*')
               ...: conn_df = fetch_simple_connections(kc_criteria, kc_criteria)
            In [1]: conn_df.head()
            Out[1]:
               bodyId_pre  bodyId_post  weight type_pre type_post instance_pre instance_post                                       conn_roiInfo
            0  1224137495   5813032771      29      KCg       KCg          KCg    KCg(super)  {'MB(R)': {'pre': 26, 'post': 26}, 'gL(R)': {'...
            1  1172713521   5813067826      27      KCg       KCg   KCg(super)         KCg-d  {'MB(R)': {'pre': 26, 'post': 26}, 'PED(R)': {...
            2   517858947   5813032943      26   KCab-p    KCab-p       KCab-p        KCab-p  {'MB(R)': {'pre': 25, 'post': 25}, 'PED(R)': {...
            3   642680826   5812980940      25   KCab-p    KCab-p       KCab-p        KCab-p  {'MB(R)': {'pre': 25, 'post': 25}, 'PED(R)': {...
            4  5813067826   1172713521      24      KCg       KCg        KCg-d    KCg(super)  {'MB(R)': {'pre': 23, 'post': 23}, 'gL(R)': {'...

            In [2]: from neuprint.utils import connection_table_to_matrix
               ...: connection_table_to_matrix(conn_df, 'type')
            Out[2]:
            type_post   KC  KCa'b'  KCab-p  KCab-sc     KCg
            type_pre
            KC           3     139       6        5     365
            KCa'b'     154  102337     245      997    1977
            KCab-p       7     310   17899     3029     127
            KCab-sc      4    2591    3975   247038    3419
            KCg        380    1969      79     1526  250351
    """
    if isinstance(group_cols, str):
        group_cols = (f"{group_cols}_pre", f"{group_cols}_post")

    assert len(group_cols) == 2, \
        "Please provide two group_cols (e.g. 'bodyId_pre', 'bodyId_post')"

    assert group_cols[0] in conn_df, \
        f"Column missing: {group_cols[0]}"

    assert group_cols[1] in conn_df, \
        f"Column missing: {group_cols[1]}"

    assert weight_col in conn_df, \
        f"Column missing: {weight_col}"

    col_pre, col_post = group_cols
    dtype = conn_df[weight_col].dtype

    agg_weights_df = conn_df.groupby([col_pre, col_post], sort=False)[weight_col].sum().reset_index()
    matrix = agg_weights_df.pivot(index=col_pre, columns=col_post, values=weight_col)
    matrix = matrix.fillna(0).astype(dtype)

    if sort_by:
        if isinstance(sort_by, str):
            sort_by = (f"{sort_by}_pre", f"{sort_by}_post")

        assert len(sort_by) == 2, \
            "Please provide two sort_by column names (e.g. 'type_pre', 'type_post')"

        pre_order = conn_df.sort_values(sort_by[0])[col_pre].unique()
        post_order = conn_df.sort_values(sort_by[1])[col_post].unique()
        matrix = matrix.reindex(index=pre_order, columns=post_order)
    else:
        # No sort: Keep the order as close to the input order as possible.
        pre_order = conn_df[col_pre].unique()
        post_order = conn_df[col_post].unique()
        matrix = matrix.reindex(index=pre_order, columns=post_order)

    if make_square:
        matrix, _ = matrix.align(matrix.T).fillna(0.0).astype(matrix.dtype)
        matrix = matrix.rename_axis('bodyId_pre', axis=0).rename_axis('bodyId_post', axis=1)
        matrix = matrix.loc[sorted(matrix.index), sorted(matrix.columns)]

    return matrix


def get_connectivity_matrix(src_ids, target_ids, src_df, target_df):
    """Fetch connectivity and return matrix grouped by L/R hemisphere."""
    print(f"Fetching connectivity: {len(src_ids)} → {len(target_ids)} neurons...")
    
    # Fetch connectivity
    neuron_df, conn_df = neu.fetch_adjacencies(src_ids, target_ids)

    # Merge properties
    conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
      
    # Convert to matrix using neuprint utility with sorting by instance (L/R)
    conn_matrix = connection_table_to_matrix(conn_df, 'bodyId', sort_by='instance')
    
    # Split by hemisphere using the sorted matrix order
    src_L, src_R, tgt_L, tgt_R = split_ids_by_side_from_matrix(conn_matrix, src_df, target_df)
    
    return conn_matrix, src_L, src_R, tgt_L, tgt_R


def get_two_hop_connectivity_matrix(src_ids, intermediate_ids, target_ids,
                                    src_df, intermediate_df, target_df,
                                    weight_method='min'):
    """
    Compute two-hop connectivity: src → intermediate → target.
    
    Parameters:
    -----------
    weight_method : str
        How to combine weights: 'min', 'product', 'second_hop', or 'count'
    """
    print(f"Computing two-hop: {len(src_ids)} → {len(intermediate_ids)} → {len(target_ids)}...")
    
    # Fetch both hops
    _, conn1 = neu.fetch_adjacencies(src_ids, intermediate_ids)  # src → intermediate
    _, conn2 = neu.fetch_adjacencies(intermediate_ids, target_ids)  # intermediate → target
    
    # Join on intermediate neurons (post from hop1 = pre from hop2)
    two_hop = conn1.merge(conn2, left_on='bodyId_post', right_on='bodyId_pre',
                          suffixes=('_hop1', '_hop2'))
    
    # Combine weights based on method
    if weight_method == 'min':
        two_hop['weight'] = two_hop[['weight_hop1', 'weight_hop2']].min(axis=1)
    elif weight_method == 'product':
        two_hop['weight'] = two_hop['weight_hop1'] * two_hop['weight_hop2']
    elif weight_method == 'second_hop':
        two_hop['weight'] = two_hop['weight_hop2']
    elif weight_method == 'count':
        two_hop['weight'] = 1
    
    # Aggregate by source and final target
    conn_agg = (two_hop.groupby(['bodyId_pre_hop1', 'bodyId_post_hop2'], as_index=False)['weight']
                .sum()
                .rename(columns={'bodyId_pre_hop1': 'bodyId_pre', 'bodyId_post_hop2': 'bodyId_post'}))
    
    print(f"  Found {len(conn_agg)} two-hop connections")
    
    # Add instance columns for sorting (merge from original dataframes)
    src_instance_map = dict(zip(src_df['bodyId'], src_df['instance']))
    tgt_instance_map = dict(zip(target_df['bodyId'], target_df['instance']))
    
    conn_agg['instance_pre'] = conn_agg['bodyId_pre'].map(src_instance_map)
    conn_agg['instance_post'] = conn_agg['bodyId_post'].map(tgt_instance_map)
    
    # Convert to matrix using neuprint utility with sorting by instance (L/R)
    conn_matrix = connection_table_to_matrix(conn_agg, 'bodyId', sort_by='instance')
    
    # Split by hemisphere using the sorted matrix order
    src_L, src_R, tgt_L, tgt_R = split_ids_by_side_from_matrix(conn_matrix, src_df, target_df)
    
    return conn_matrix, src_L, src_R, tgt_L, tgt_R


def cluster_connectivity_matrix(conn_matrix, method='cosine', linkage_method='average', min_connections=1):
    """
    Cluster rows and columns of connectivity matrix by similarity.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to cluster
    method : str
        Distance metric: 'cosine', 'correlation', 'euclidean'
    linkage_method : str
        Linkage method: 'average', 'ward', 'complete', 'single'
    min_connections : int
        Minimum total connections to include neuron in clustering
    
    Returns:
    --------
    clustered_matrix : pd.DataFrame
        Reordered matrix
    row_linkage : ndarray
        Hierarchical clustering linkage for rows
    col_linkage : ndarray
        Hierarchical clustering linkage for columns
    """
    # Filter out neurons with too few connections
    row_sums = conn_matrix.sum(axis=1)
    col_sums = conn_matrix.sum(axis=0)
    
    valid_rows = row_sums >= min_connections
    valid_cols = col_sums >= min_connections
    
    filtered_matrix = conn_matrix.loc[valid_rows, valid_cols]
    
    if len(filtered_matrix) == 0:
        print("Warning: No neurons pass the minimum connection threshold!")
        return conn_matrix, None, None
    
    print(f"Clustering {valid_rows.sum()}/{len(conn_matrix)} sources, {valid_cols.sum()}/{len(conn_matrix.columns)} targets")
    
    # Compute distance and cluster
    def safe_pdist(data, metric):
        """Compute pdist and handle NaN/Inf values."""
        dist = pdist(data, metric=metric)
        # Replace any NaN or Inf with max distance
        if not np.all(np.isfinite(dist)):
            max_dist = np.nanmax(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 1.0
            dist = np.nan_to_num(dist, nan=max_dist, posinf=max_dist, neginf=0)
        return dist
    
    if method == 'cosine':
        row_dist = safe_pdist(filtered_matrix.values, metric='cosine')
        col_dist = safe_pdist(filtered_matrix.T.values, metric='cosine')
    elif method == 'correlation':
        row_dist = safe_pdist(filtered_matrix.values, metric='correlation')
        col_dist = safe_pdist(filtered_matrix.T.values, metric='correlation')
    else:
        row_dist = safe_pdist(filtered_matrix.values, metric=method)
        col_dist = safe_pdist(filtered_matrix.T.values, metric=method)
    
    row_linkage = linkage(row_dist, method=linkage_method)
    col_linkage = linkage(col_dist, method=linkage_method)
    
    # Get reordered indices
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    
    # Reorder matrix
    clustered_matrix = filtered_matrix.iloc[row_order, col_order]
    
    return clustered_matrix, row_linkage, col_linkage


def matmul_conn_matrices(conn_df1, conn_df2, weight_label='weight',
                         sort_rows=None, sort_cols=None,
                         conn1_pre='type_pre', conn1_post='instance_post',
                         conn2_pre='instance_pre', conn2_post='type_post',
                         return_all=False):
    '''
    Multiply two connection matrices.
    Args:
        conn_df1: DataFrame, the first connection dataframe
        conn_df2: DataFrame, the second connection dataframe
        weight_label: str, the column to use for the weight
        sort_rows: str or list, how to sort the inputs 
        sort_cols: str or list, how to sort the outputs
        conn1_pre: str, the pre column for the first connection dataframe
        conn1_post: str, the post column for the first connection dataframe
        conn2_pre: str, the pre column for the second connection dataframe  
        conn2_post: str, the post column for the second connection dataframe
        return_all: bool, whether to return all three matrices
    Returns:
        conn_combined: DataFrame, the combined connection matrix
    '''
    # Get connection matrices
    conn_matrix1 = connection_table_to_matrix(conn_df1,
                        group_cols=[conn1_pre, conn1_post],
                        sort_by= ['weight', 'weight'],
                        weight_col=weight_label)
    conn_matrix2 = connection_table_to_matrix(conn_df2,
                        group_cols=[conn2_pre, conn2_post],
                        sort_by= ['weight', 'weight'],
                        weight_col=weight_label)
    # Sort labels
    intermediate_neurons = conn_df1[conn1_post].unique()
    conn_matrix1 = sort_matrix_labels(conn_matrix1, conn_df=conn_df1, 
                                       sort_rows_by=sort_rows, 
                                       sort_cols_by=intermediate_neurons)
    
    conn_matrix2 = sort_matrix_labels(conn_matrix2, conn_df=conn_df2, 
                                       sort_rows_by=intermediate_neurons, 
                                       sort_cols_by=sort_cols)
    conn_combined = conn_matrix1.dot(conn_matrix2)
    if return_all:
        return conn_matrix1, conn_matrix2, conn_combined
    else:
        return conn_combined


# ============================== CONNECTION MATRIX PLOTTING FUNCTIONS ==============================
def plot_connection_matrix(conn_matrix, 
                           show_all_row_labels=False,
                           show_all_col_labels=False,
                           col_label_interval=None,
                           row_label_interval=None,
                           normalize_colors=True,
                           vmin=10, vmax=None,
                           cbar_shrink=0.2,
                           colorbar_label='weight',
                           figsize=None,
                           show_grid=False,
                           grid_lw=0.5,
                           grid_color='white',
                           ax=None, min_fontsize=6):
    '''
    Plot a connection matrix.
    Args:
        conn_matrix: DataFrame, the connection matrix to plot
        show_all_row_labels: bool, whether to show all row labels
        show_all_col_labels: bool, whether to show all column labels
        normalize_colors: bool, whether to normalize the colors
        vmin: float, the minimum value for the colorbar
        vmax: float, the maximum value for the colorbar
        colorbar_label: str, the label for the colorbar
        figsize: tuple, the size of the figure
        show_grid: bool, whether to show the grid
        grid_lw: float, the width of the grid lines
        grid_color: str, the color of the grid lines
        ax: matplotlib axis, the axis to plot on
    Returns:
        fig: matplotlib figure, the figure with the plot
    '''
    n_rows = len(conn_matrix.index)
    n_cols = len(conn_matrix.columns)
    if ax is None:
        # Auto-adjust figure size if showing all labels
        if figsize is None:
            if show_all_row_labels or show_all_col_labels:
                # Calculate size based on number of labels
                height = max(10, n_rows * 0.3) if show_all_row_labels else 10
                width = max(10, n_cols * 0.3) if show_all_col_labels else 10
                figsize = (width, height)
            else:
                figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # make colorbar smaller to leave room for roi legend
    # control which tick labels are shown
    yticklabels = True if show_all_row_labels else 'auto'
    xticklabels = True if show_all_col_labels else 'auto'
    
    # Set color limits based on normalization option
    if normalize_colors and vmax is None:
        # Use the actual data range for better color contrast
        vmin = conn_matrix.min().min() if not conn_matrix.empty else 0
        vmax = conn_matrix.max().max() if not conn_matrix.empty else 1
        # Skip very low values to improve contrast
        if vmin < 0.1 * vmax:
            vmin = 0.1 * vmax
    else:
        vmin = vmin
        vmax = vmax
    
    # Create heatmap
    sns.heatmap(conn_matrix, ax=ax, vmin=vmin, vmax=vmax, cmap='magma',
                cbar_kws={'shrink': cbar_shrink, 'anchor': (0, 0.0), 'label': colorbar_label},
                yticklabels=yticklabels, xticklabels=xticklabels,
                linewidths=grid_lw if show_grid else 0,
                linecolor=grid_color if show_grid else None)

    # Adjust tick labels to show every Nth label if interval is specified
    if show_all_row_labels and row_label_interval is not None and row_label_interval > 1:
        yticklabels = ax.get_yticklabels()
        for i, label in enumerate(yticklabels):
            if i % row_label_interval != 0:
                label.set_text('')
        ax.set_yticklabels([label.get_text() for label in yticklabels])
    
    if show_all_col_labels and col_label_interval is not None and col_label_interval > 1:
        xticklabels = ax.get_xticklabels()
        for i, label in enumerate(xticklabels):
            if i % col_label_interval != 0:
                label.set_text('')
        ax.set_xticklabels([label.get_text() for label in xticklabels])
    
   
    # Add complete border if grid is enabled
    if show_grid:
        n_rows, n_cols = conn_matrix.shape
        # Get the current limits of the heatmap
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
       
        sns.despine(ax=ax, right=False, bottom=False)
       
    # Modify colorbar tick labels if vmax is provided and not 1.0
    if vmax is not None and vmax != 1.0:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            # Get current tick positions and labels
            ticks = cbar.ax.get_yticks()
            tick_labels = cbar.ax.get_yticklabels()
            if tick_labels and len(ticks) > 0:
                # Get max tick_label value
                print("Replacing tick labels")
                max_tick_label = max([float(label.get_text()) for label in tick_labels])
                print(max_tick_label)
                
                # Create new label list
                new_labels = [label.get_text() for label in tick_labels]
                new_labels[-1] = f'>{vmax}' #f'>{vmax:.2f}'
                
                # Use FixedLocator to force all ticks to be visible (prevents auto-hiding)
                from matplotlib.ticker import FixedLocator
                cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
                cbar.ax.set_yticklabels(new_labels)
        else:
            print("No colorbar found")
    # Adjust tick label font size when showing all labels
    #if show_all_row_labels:
    ax.tick_params(axis='y', labelsize=min_fontsize)
    #if show_all_col_labels:
    ax.tick_params(axis='x', labelsize=min_fontsize, rotation=90)

    return fig


def plot_grouped_connection_matrix(conn_matrix, conn_df, 
                                   pre_grouper_dict=None, 
                                   post_grouper_dict=None,
                                   sorted_by_grouper=False, 
                                   group_per_row=None,
                                   group_per_col=None,
                                   pre_variable='type_pre',
                                   post_variable='type_post',
                                   post_grouper='type_post',
                                   pre_grouper='type_pre',
                                   annotate_rows=True,
                                   annotate_cols=False,
                                   show_all_row_labels=False,
                                   show_all_col_labels=False,
                                   row_label_interval=None,
                                   col_label_interval=None,
                                   figsize=None,
                                   normalize_colors=True,
                                   vmin=None, vmax=None,
                                   colorbar_label='weight',
                                   colorbar_shrink=0.2,
                                   ax=None, min_fontsize=6):
    """
    Plot connection matrix with ROI group annotations.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to plot
    conn_df : pd.DataFrame
        Connection dataframe with ROI and type information
    roi_dict : dict, optional
        Dictionary mapping ROI names to colors
    sorted_by_grouper : bool, default False
        If True, data is sorted by grouper so show colored line blocks. If False, color tick labels.
    pre_variable : str, default 'type_pre'
        Column name for row grouping variable
    post_variable : str, default 'type_post'
        Column name for column grouping variable
    annotate_rows : bool, default True
        Whether to annotate rows (inputs)
    annotate_cols : bool, default False
        Whether to annotate columns (outputs)
    show_all_row_labels : bool, default False
        If True, show all row tick labels (or every N if row_label_interval is set)
    show_all_col_labels : bool, default False
        If True, show all column tick labels (or every N if col_label_interval is set)
    row_label_interval : int, optional
        If show_all_row_labels is True, show every Nth row label (e.g., 5 for every 5th label)
    col_label_interval : int, optional
        If show_all_col_labels is True, show every Nth column label (e.g., 5 for every 5th label)
    figsize : tuple, optional
        Figure size (width, height). If None, uses (10, 10) or auto-adjusts for all labels
    normalize_colors : bool, default True
        If True, normalize the colorbar to use the full range of the data
    vmin : float, optional
        Minimum value for the colorbar
    vmax : float, optional
        Maximum value for the colorbar
    colorbar_label : str, default 'weight'
        Label for the colorbar
    colorbar_shrink : float, default 0.2
        Shrink factor for the colorbar
    ax : matplotlib axis, optional
        Axis to plot on
    """
    n_rows = len(conn_matrix.index)
    n_cols = len(conn_matrix.columns)

    if ax is None:
        # Auto-adjust figure size if showing all labels
        if figsize is None:
            if show_all_row_labels or show_all_col_labels:
                # Calculate size based on number of labels
                height = max(10, n_rows * 0.3) if show_all_row_labels else 10
                width = max(10, n_cols * 0.3) if show_all_col_labels else 10
                figsize = (width, height)
            else:
                figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
       
    if pre_grouper_dict is None:
        pre_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(conn_df[pre_grouper].unique())}
    if post_grouper_dict is None:
        post_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(conn_df[post_grouper].unique())}
        
    # make colorbar smaller to leave room for roi legend
    # control which tick labels are shown
    yticklabels = True if show_all_row_labels else 'auto'
    xticklabels = True if show_all_col_labels else 'auto'
    
    # Set color limits based on normalization option
    if normalize_colors:
        if vmin is None:
            # Use the actual data range for better color contrast
            vmin = conn_matrix.min().min() if not conn_matrix.empty else 0
        if vmax is None:
            vmax = conn_matrix.max().max() if not conn_matrix.empty else 1
        # Skip very low values to improve contrast
        if vmin < 0.1 * vmax:
            vmin = 0.1 * vmax
    else:
        vmin = vmin
        vmax = None if vmax is None else vmax
    
    sns.heatmap(conn_matrix, ax=ax, vmin=vmin, vmax=vmax, cmap='magma',
                cbar_kws={'shrink': colorbar_shrink, 'anchor': (0, 0.0), 'label': colorbar_label},
                yticklabels=yticklabels, xticklabels=xticklabels)
    
    # Adjust tick labels to show every Nth label if interval is specified
    if show_all_row_labels and row_label_interval is not None and row_label_interval > 1:
        yticklabels = ax.get_yticklabels()
        for i, label in enumerate(yticklabels):
            if i % row_label_interval != 0:
                label.set_text('')
        ax.set_yticklabels([label.get_text() for label in yticklabels])
    
    if show_all_col_labels and col_label_interval is not None and col_label_interval > 1:
        xticklabels = ax.get_xticklabels()
        for i, label in enumerate(xticklabels):
            if i % col_label_interval != 0:
                label.set_text('')
        ax.set_xticklabels([label.get_text() for label in xticklabels])
    
    # Adjust tick label font size when showing all labels
    #if show_all_row_labels:
    ax.tick_params(axis='y', labelsize=min_fontsize)
    #if show_all_col_labels:
    ax.tick_params(axis='x', labelsize=min_fontsize, rotation=90)

    from matplotlib.patches import Patch
    
    # Annotate rows (inputs/pre)
    if annotate_rows:
        # Get ROI for each row in the connection matrix
        if group_per_row is None:
            group_per_row = [conn_df[conn_df[pre_variable]==bodyId][pre_grouper].values[0] 
                    for bodyId in conn_matrix.index.tolist()]
            print(group_per_row)
        
        
        if sorted_by_grouper:
            # Option 1: Add vertical line blocks on left to show ROI groups
            roi_boundaries = [0]
            for i in range(1, len(group_per_row)):
                if group_per_row[i] != group_per_row[i-1]:
                    roi_boundaries.append(i)
            roi_boundaries.append(len(group_per_row))
            
            # Calculate relative positions based on actual plot dimensions
            xlim = ax.get_xlim()
            plot_width = xlim[1] - xlim[0]
            line_offset = xlim[0] - 0.01 * plot_width  # 1% of plot width to the left
            label_offset = xlim[0] - 0.02 * plot_width  # 3% of plot width to the left
            
            # Draw vertical lines spanning each ROI group and add labels on the left
            for i in range(len(roi_boundaries)-1):
                y_start = roi_boundaries[i]
                y_end = roi_boundaries[i+1]
                roi_name = group_per_row[roi_boundaries[i]]
                roi_color = pre_grouper_dict[roi_name]
                
                # Draw vertical line spanning this ROI group (offset from plot)
                ax.plot([line_offset, line_offset], [y_start, y_end], 
                        color=roi_color, linewidth=4, solid_capstyle='butt', 
                        clip_on=False)
                
                # Add ROI label on the left side, centered vertically with vertical orientation
                y_center = (y_start + y_end) / 2
                ax.text(label_offset, y_center, roi_name, 
                        va='center', ha='center', fontsize=10, 
                        color=roi_color, 
                        rotation=90, clip_on=False)
            
            # Offset y-tick labels to avoid overlap with ROI annotations
            ax.tick_params(axis='y', pad=40)
        else:
            # Option 2: Color individual tick labels by ROI
            yticklabels = ax.get_yticklabels()
            for idx, (label, roi) in enumerate(zip(yticklabels, group_per_row)):
                label.set_color(pre_grouper_dict[roi])
            
            # Add color legend for ROI groups (positioned above the colorbar)
            legend_elements = [Patch(facecolor=pre_grouper_dict[roi], label=roi) 
                            for roi in pre_grouper_dict.keys()]
            ax.legend(handles=legend_elements, loc='upper left', 
                    bbox_to_anchor=(1.05, 1.0), frameon=True, title='ROI')
    
    # Annotate columns (outputs/post)
    if annotate_cols:
        # Get ROI for each column in the connection matrix
        if group_per_col is None:
            group_per_col = [conn_df[conn_df[post_variable]==bodyId][post_grouper].values[0] 
                      for bodyId in conn_matrix.columns.tolist()]
        
        if sorted_by_grouper:
            # Option 1: Add horizontal line blocks on bottom to show ROI groups
            roi_boundaries = [0]
            for i in range(1, len(group_per_col)):
                if group_per_col[i] != group_per_col[i-1]:
                    roi_boundaries.append(i)
            roi_boundaries.append(len(group_per_col))
            
            # Calculate relative positions based on actual plot dimensions
            # For column annotations, we want them just below the x-axis tick labels
            ylim = ax.get_ylim()
            plot_height = ylim[1] - ylim[0]
            line_offset = ylim[1] - 0.01 * plot_height  # 2% of plot height below
            label_offset = ylim[1] - 0.02 * plot_height  # 5% of plot height below
            
            # Draw horizontal lines spanning each ROI group and add labels on bottom
            for i in range(len(roi_boundaries)-1):
                x_start = roi_boundaries[i]
                x_end = roi_boundaries[i+1]
                roi_name = group_per_col[roi_boundaries[i]]
                roi_color = post_grouper_dict[roi_name]
                
                # Draw horizontal line spanning this ROI group (offset from plot at bottom)
                ax.plot([x_start, x_end], [n_rows + line_offset, n_rows + line_offset], 
                        color=roi_color, linewidth=4, solid_capstyle='butt', 
                        clip_on=False)
                
                # Add ROI label on the bottom, centered horizontally
                x_center = (x_start + x_end) / 2
                ax.text(x_center, n_rows + label_offset, roi_name, 
                        va='center', ha='center', fontsize=10, 
                        color=roi_color, 
                        rotation=0, clip_on=False)
            
            # Offset x-tick labels to avoid overlap with ROI annotations
            ax.tick_params(axis='x', pad=40)
        else:
            # Option 2: Color individual tick labels by ROI
            xticklabels = ax.get_xticklabels()
            for idx, (label, roi) in enumerate(zip(xticklabels, group_per_col)):
                label.set_color(post_grouper_dict[roi])

    # Modify colorbar tick labels if vmax is provided and not 1.0
    if vmax is not None and vmax != 1.0:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            # Get current tick positions and labels
            ticks = cbar.ax.get_yticks()
            tick_labels = cbar.ax.get_yticklabels()
            if tick_labels and len(ticks) > 0:
                # Get max tick_label value
                print(f"Replacing tick labels (vmax={vmax})")
                max_tick_label = max([float(label.get_text()) for label in tick_labels])
                print(max_tick_label)
                
                # Create new label list
                new_labels = [label.get_text() for label in tick_labels]
                new_labels[-1] = f'>{vmax}' #f'>{vmax:.2f}'
                
                # Use FixedLocator to force all ticks to be visible (prevents auto-hiding)
                from matplotlib.ticker import FixedLocator
                cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
                cbar.ax.set_yticklabels(new_labels)
                print(new_labels)
        else:
            print("No colorbar found")

    return fig


def highlight_row_or_column(ax, conn_matrix, row_label=None, column_label=None, 
                           color='white', linewidth=2, 
                           highlight_label=True, highlight_label_color='red',
                           highlight_box=True):
    """
    Draw a thin box around a specific row or column in a connection matrix plot.
    If multiple rows/columns have the same label, highlights all of them.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis containing the heatmap
    conn_matrix : pd.DataFrame
        The connection matrix used for the heatmap
    row_label : str or list, optional
        Label(s) of the row(s) to highlight. Can be a single label or list of labels.
    column_label : str or list, optional
        Label(s) of the column(s) to highlight. Can be a single label or list of labels.
    color : str, default 'white'
        Color of the highlight box
    linewidth : float, default 2
        Width of the highlight box lines
    highlight_label : bool, default True
        Whether to add asterisks to the highlighted tick labels
    highlight_label_color : str, default 'red'
        Color of the asterisks added to highlighted labels
    """
    from matplotlib.patches import Rectangle
    
    if row_label is None and column_label is None:
        raise ValueError("Either row_label or column_label must be specified")
    if row_label is not None and column_label is not None:
        raise ValueError("Only one of row_label or column_label can be specified")
    
    if row_label is not None:
        # Handle both single labels and lists of labels
        if isinstance(row_label, str):
            row_labels = [row_label]
        else:
            row_labels = row_label
        
        # Check that all labels exist
        for label in row_labels:
            if label not in conn_matrix.index:
                raise ValueError(f"Row label '{label}' not found in matrix index")
        
        # Find all positions where any of the row labels appear
        all_row_positions = []
        for label in row_labels:
            row_mask = conn_matrix.index == label
            row_positions = np.where(row_mask)[0]
            all_row_positions.extend(row_positions)
        
        # Create rectangles for all matching rows and add asterisks to labels
        for row_pos in all_row_positions:
            rect = Rectangle((0, row_pos), len(conn_matrix.columns), 1,
                            linewidth=linewidth, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add asterisk to the row label if requested
            if highlight_label:
                yticklabels = ax.get_yticklabels()
                if row_pos < len(yticklabels) and yticklabels[row_pos].get_text():
                    original_text = yticklabels[row_pos].get_text()
                    if not original_text.endswith('*'):  # Avoid adding multiple asterisks
                        yticklabels[row_pos].set_text(f"{original_text}*")
                        yticklabels[row_pos].set_color(highlight_label_color)  # Use custom color
    
    if column_label is not None:
        # Handle both single labels and lists of labels
        if isinstance(column_label, str):
            column_labels = [column_label]
        else:
            column_labels = column_label
        
        # Check that all labels exist
        for label in column_labels:
            if label not in conn_matrix.columns:
                raise ValueError(f"Column label '{label}' not found in matrix columns")
        
        # Find all positions where any of the column labels appear
        all_col_positions = []
        for label in column_labels:
            col_mask = conn_matrix.columns == label
            col_positions = np.where(col_mask)[0]
            all_col_positions.extend(col_positions)
        
        # Create rectangles for all matching columns and add asterisks to labels
        for col_pos in all_col_positions:
            if highlight_box:
                rect = Rectangle((col_pos, 0), 1, len(conn_matrix.index),
                                linewidth=linewidth, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            
            # Add asterisk to the column label if requested
            if highlight_label:
                xticklabels = ax.get_xticklabels()
                if col_pos < len(xticklabels) and xticklabels[col_pos].get_text():
                    original_text = xticklabels[col_pos].get_text()
                    if not original_text.endswith('*'):  # Avoid adding multiple asterisks
                        xticklabels[col_pos].set_text(f"{original_text}*")
                        xticklabels[col_pos].set_color(highlight_label_color)  # Use custom color


#%%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import leaves_list

def hier_cosine(indata,distance_thresh):
    '''
    From CDowell, compare with combined row+col clustering
    Compute cosine similarity between all pairs of rows in the input matrix.
   
    Usage:
        rows = mat_to_cluster.index.tolist()
        cols = mat_to_cluster.columns.tolist()
        mat_to_cluster = mat_to_cluster.values

        cluster, dmat = hier_cosine( mat_to_cluster, distance_thresh=1)
        z = linkage_order(cluster)

        # Reorder the matrix based on clustering
        print(len(z), mat_to_cluster.shape)
        clustered_mat = mat_to_cluster[z, :] 

        clustered_mat = pd.DataFrame(clustered_mat, index=rows, columns=cols)

    Args:
        indata: Input matrix
        distance_thresh: Distance threshold for clustering (0=perfectly similar, 1=orthogonal)
        Lower numbers, more strict (only merges vv similar clusters), distance_thresh=1, stops merging when clusters somewhat dissimilar
        
    Returns:
        cluster: Clustering model
        d_mat: Distance matrix
    '''
    
    in_shape = np.shape(indata)
    # Create similarity matrix, n_rows x n_rows
    # in_shape[0] = rows in matrix (number of pre-syn types, for ex.)
    sim_mat = np.zeros([in_shape[0], in_shape[0]],dtype='float64')
    ilen = int(in_shape[0])
    for i in range(in_shape[0]): # loop over rows
        x = indata[i,:] # Take entire ROW i
        for z in range(in_shape[0]):
            y = indata[z,:] # Take entire ROW z
            sim_mat[i,z] = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
            if np.isnan(sim_mat[i,z]):
                print('i',i)
                print('z',z)
    d_mat = 1-sim_mat

    cluster = AgglomerativeClustering(metric='precomputed', linkage='single', 
                                    compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
    cluster.fit(d_mat)
    return cluster, d_mat

def linkage_order(model):
    ''' From CDowell, compare with combined row+col clustering
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    z = leaves_list(linkage_matrix)
    return z


def _apply_thresholding(conn_matrix, threshold=None, threshold_percentile=90):
    """
    Apply thresholding to connection matrix.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to threshold
    threshold : float, optional
        Absolute threshold for connections
    threshold_percentile : float, default 90
        Percentile threshold (keep top X% of connections per row/column)
    
    Returns:
    --------
    matrix_thresholded : pd.DataFrame
        Thresholded matrix
    """
    if threshold is not None:
        # Absolute threshold
        return conn_matrix.where(conn_matrix > threshold, 0)
    else:
        # Percentile-based thresholding
        matrix_thresholded = conn_matrix.copy()
        for i in range(len(conn_matrix)):
            # Keep only top percentile of connections per row
            threshold_val = np.percentile(conn_matrix.iloc[i].values, threshold_percentile)
            matrix_thresholded.iloc[i] = conn_matrix.iloc[i].where(
                conn_matrix.iloc[i] >= threshold_val, 0)
        
        for j in range(len(conn_matrix.columns)):
            # Keep only top percentile of connections per column
            threshold_val = np.percentile(conn_matrix.iloc[:, j].values, threshold_percentile)
            matrix_thresholded.iloc[:, j] = conn_matrix.iloc[:, j].where(
                conn_matrix.iloc[:, j] >= threshold_val, 0)
        
        return matrix_thresholded


def _compute_clustering(matrix_filled, method='ward'):
    """
    Compute hierarchical clustering using cosine similarity.
    
    Parameters:
    -----------
    matrix_filled : pd.DataFrame or np.ndarray
        Matrix with NaN values filled (thresholded or original)
    method : str, default 'ward'
        Linkage method for hierarchical clustering
    
    Returns:
    --------
    row_linkage : np.ndarray
        Linkage matrix for rows
    col_linkage : np.ndarray
        Linkage matrix for columns
    """
    # Calculate cosine similarity for rows (source patterns)
    row_similarity = cosine_similarity(matrix_filled.values)
    # Handle NaN values first (can occur when rows have zero norm) - set to 0 (max distance)
    row_similarity = np.nan_to_num(row_similarity, nan=0.0)
    # Clip similarity to [0, 1] to handle numerical errors and ensure non-negative distances
    row_similarity = np.clip(row_similarity, 0, 1)
    row_distance = 1 - row_similarity  # Convert similarity to distance
    np.fill_diagonal(row_distance, 0)  # Ensure diagonal is exactly zero
    # Ensure all distances are non-negative (handle any remaining numerical errors)
    row_distance = np.maximum(row_distance, 0)
    row_linkage = linkage(squareform(row_distance), method=method)
    
    # Calculate cosine similarity for columns (target patterns)
    col_similarity = cosine_similarity(matrix_filled.values.T)
    # Handle NaN values first (can occur when columns have zero norm) - set to 0 (max distance)
    col_similarity = np.nan_to_num(col_similarity, nan=0.0)
    # Clip similarity to [0, 1] to handle numerical errors and ensure non-negative distances
    col_similarity = np.clip(col_similarity, 0, 1)
    col_distance = 1 - col_similarity  # Convert similarity to distance
    np.fill_diagonal(col_distance, 0)  # Ensure diagonal is exactly zero
    # Ensure all distances are non-negative (handle any remaining numerical errors)
    col_distance = np.maximum(col_distance, 0)
    col_linkage = linkage(squareform(col_distance), method=method)
    
    return row_linkage, col_linkage


def cluster_matrix_cosine_similarity(conn_matrix, method='ward', 
                                     threshold=None, threshold_percentile=None):
    """
    Cluster connection matrix using cosine similarity and hierarchical clustering.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to cluster
    method : str, default 'ward'
        Linkage method for hierarchical clustering ('ward', 'complete', 'average', 'single')
    threshold : float, optional
        Absolute threshold for connections (e.g., 0.05). Higher values keeps most connections (conservative, 90). Lower (e.g., 50) keeps only strongest connections.
    threshold_percentile : float, optional
        Percentile threshold (keep top X% of connections per row/column). If provided, overrides threshold.
    
    Returns:
    --------
    clustered_matrix : pd.DataFrame
        Reordered matrix based on clustering
    row_linkage : np.ndarray
        Linkage matrix for rows
    col_linkage : np.ndarray
        Linkage matrix for columns
    matrix_used : pd.DataFrame, optional
        Thresholded matrix used for clustering (only returned if thresholding was applied)
    """
    # Apply thresholding if specified
    if threshold is not None or threshold_percentile is not None:
        matrix_used = _apply_thresholding(conn_matrix, threshold, threshold_percentile)
        matrix_filled = matrix_used.fillna(0)
        return_thresholded = True
    else:
        matrix_filled = conn_matrix.fillna(0)
        return_thresholded = False
    
    # Compute clustering
    row_linkage, col_linkage = _compute_clustering(matrix_filled, method)
    
    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    
    # Reorder the original matrix (not the thresholded one)
    clustered_matrix = conn_matrix.iloc[row_order, col_order]
    
    if return_thresholded:
        return clustered_matrix, row_linkage, col_linkage, matrix_used
    else:
        return clustered_matrix, row_linkage, col_linkage


def cluster_matrix_cosine_similarity_thresholded(conn_matrix, method='ward', 
                                                 threshold=None, threshold_percentile=90):
    """
    Convenience function for thresholded clustering. Calls cluster_matrix_cosine_similarity with thresholding.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to cluster
    method : str, default 'ward'
        Linkage method for hierarchical clustering
    threshold : float, optional
        Absolute threshold for connections (e.g., 0.05).
        Higher values keeps most connections (conservative, 90). 
        Lower (e.g., 50) keeps only strongest connections.
    threshold_percentile : float, default 90
        Percentile threshold (keep top X% of connections per row/column)
    
    Returns:
    --------
    clustered_matrix : pd.DataFrame
        Reordered matrix based on clustering
    row_linkage : np.ndarray
        Linkage matrix for rows
    col_linkage : np.ndarray
        Linkage matrix for columns
    matrix_thresholded : pd.DataFrame
        Thresholded matrix used for clustering
    """
    return cluster_matrix_cosine_similarity(conn_matrix, method, threshold, threshold_percentile)


def plot_dendrograms(row_linkage, col_linkage, row_labels=None, col_labels=None, 
                    figsize=(15, 8), n_clusters=None):
    """
    Plot dendrograms for row and column clustering.
    
    Parameters:
    -----------
    row_linkage : np.ndarray
        Linkage matrix for rows
    col_linkage : np.ndarray
        Linkage matrix for columns
    row_labels : list, optional
        Labels for rows
    col_labels : list, optional
        Labels for columns
    figsize : tuple, default (15, 8)
        Figure size
    n_clusters : int, optional
        Number of clusters to highlight with colors
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot row dendrogram: which types are most/least similar?
    dendrogram(row_linkage, ax=ax1, orientation='left', labels=row_labels,
               color_threshold=0.7*np.max(row_linkage[:,2]) if n_clusters is None else None)
    ax1.set_title('Source Clustering (Output Patterns)')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('P1 Types (Pre-synaptic Sources)')
    
    # Plot column dendrogram
    dendrogram(col_linkage, ax=ax2, orientation='top', labels=col_labels,
               color_threshold=0.7*np.max(col_linkage[:,2]) if n_clusters is None else None)
    ax2.set_title('Target Clustering (Input Patterns)')
    ax2.set_xlabel('P1 Types (Post-synaptic Targets)')
    ax2.set_ylabel('Distance')
    
    plt.tight_layout()
    return fig


def plot_cluster_analysis(clustered_matrix, row_linkage, col_linkage, 
                         n_clusters=5, figsize=(20, 12), grid_lw=0, 
                         show_all_labels=False, label_fontsize=8):
    """
    Create a comprehensive cluster analysis plot with dendrograms and heatmap.
    
    Parameters:
    -----------
    clustered_matrix : pd.DataFrame
        Clustered connection matrix
    row_linkage : np.ndarray
        Linkage matrix for rows
    col_linkage : np.ndarray
        Linkage matrix for columns
    n_clusters : int, default 5
        Number of clusters to analyze
    figsize : tuple, default (20, 12)
        Figure size
    grid_lw : float, default 0
        Grid line width for heatmap
    show_all_labels : bool, default False
        Whether to show all row and column labels
    label_fontsize : int, default 8
        Font size for labels when show_all_labels=True
    """
    from scipy.cluster.hierarchy import fcluster
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 4, 1],
                         hspace=0.3, wspace=0.4)
    
    # Prepare labels for dendrograms and heatmap
    if show_all_labels:
        row_labels = clustered_matrix.index.tolist()
        col_labels = clustered_matrix.columns.tolist()
        yticklabels = True
        xticklabels = True
    else:
        row_labels = None
        col_labels = None
        yticklabels = True
        xticklabels = True
    
    # Row dendrogram (left)
    ax_row_dendro = fig.add_subplot(gs[:, 0])
    dendrogram(row_linkage, ax=ax_row_dendro, orientation='left',
               color_threshold=0.7*np.max(row_linkage[:,2]),
               labels=row_labels)
    ax_row_dendro.set_title('Source\nClusters', fontsize=12, pad=10)
    ax_row_dendro.set_xlabel('Distance')
    if show_all_labels:
        ax_row_dendro.tick_params(axis='y', labelsize=label_fontsize)
    
    # Column dendrogram (top)
    ax_col_dendro = fig.add_subplot(gs[0, 1])
    dendrogram(col_linkage, ax=ax_col_dendro, orientation='top',
               color_threshold=0.7*np.max(col_linkage[:,2]),
               labels=col_labels)
    ax_col_dendro.set_title('Target Clusters', fontsize=12, pad=10)
    ax_col_dendro.set_ylabel('Distance')
    if show_all_labels:
        ax_col_dendro.tick_params(axis='x', labelsize=label_fontsize, rotation=90)
    
    # Main heatmap (center)
    ax_heatmap = fig.add_subplot(gs[1:, 1])
    
    # Plot clustered heatmap
    sns.heatmap(clustered_matrix, ax=ax_heatmap, cmap='viridis', 
                cbar_kws={'shrink': 0.8}, linewidths=grid_lw,
                yticklabels=yticklabels, xticklabels=xticklabels)
    ax_heatmap.set_title('P1-P1 Connections (Cosine Similarity Clustered)', fontsize=12, pad=10)
    ax_heatmap.set_xlabel('Post-synaptic P1 Type')
    ax_heatmap.set_ylabel('Pre-synaptic P1 Type')
    
    # Adjust label formatting for heatmap
    if show_all_labels:
        ax_heatmap.tick_params(axis='both', labelsize=label_fontsize)
        # Rotate x-axis labels for better readability
        plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add cluster information
    row_clusters = fcluster(row_linkage, n_clusters, criterion='maxclust')
    col_clusters = fcluster(col_linkage, n_clusters, criterion='maxclust')
    
    # Print cluster information
    print(f"\nRow Clusters (n={n_clusters}):")
    for i in range(1, n_clusters + 1):
        cluster_mask = row_clusters == i
        cluster_labels = clustered_matrix.index[cluster_mask].tolist()
        print(f"  Cluster {i}: {cluster_labels}")
    
    print(f"\nColumn Clusters (n={n_clusters}):")
    for i in range(1, n_clusters + 1):
        cluster_mask = col_clusters == i
        cluster_labels = clustered_matrix.columns[cluster_mask].tolist()
        print(f"  Cluster {i}: {cluster_labels}")
    
    return fig, row_clusters, col_clusters



# %%
