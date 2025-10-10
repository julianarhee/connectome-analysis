#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee / Rishika Mohanta
 # @ Email: juliana.rhee@gmail.com
 # @ Create Time: 2025-03-20 11:04:26
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-03-26 10:52:42
 # @ Description: A set of functions to analyze connectivity data. 
 # 
 # Base connectivity (Nth order) funcs from Rishika.
 '''

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# For community detection using the Louvain algorithm
#import community as community_louvain  
# For network embedding using node2vec
#from node2vec import Node2Vec
# For visualization of the embedding use umap
#from umap import UMAP

#%%


def network_influence_propagate(input_root_ids, input_values, exclude_downstream=None,
                                start_pre_post='pre_root_id', target_pre_post='post_root_id'):
    # find all downstream connections
    downstream_connections = connections[connections[start_pre_post].isin(input_root_ids)].copy().reset_index(drop=True)
    # remove connections with invalid synapse strength
    downstream_connections = downstream_connections[np.logical_not(np.isnan(downstream_connections['syn_strength']))]
    downstream_connections = downstream_connections[np.logical_and(downstream_connections['syn_strength'] > 0, downstream_connections['syn_strength'] < 1)]
    if exclude_downstream is not None:
        # remove excluded downstream neurons
        downstream_connections = downstream_connections[np.logical_not(downstream_connections['post_root_id'].isin(exclude_downstream))]
    # find all downstream neurons
    downstream_neurons = np.unique(downstream_connections[target_pre_post])
    # create a mapping from root id to index
    root_to_input = {r:i for i,r in enumerate(input_root_ids)}
    root_to_output = {r:i for i,r in enumerate(downstream_neurons)}
    # create a connectivity matrix
    connection_matrix = np.zeros((len(downstream_neurons), len(input_root_ids)))
    connection_counts = np.zeros((len(downstream_neurons), len(input_root_ids)))
    # fill the matrix
    for i in tqdm(downstream_connections.index):
        pre_id = downstream_connections.loc[i, start_pre_post]
        post_id = downstream_connections.loc[i, target_pre_post]
        pre_index = root_to_input[pre_id]
        post_index = root_to_output[post_id]
        if connection_matrix[post_index, pre_index] == 0:
            connection_matrix[post_index, pre_index] = downstream_connections.loc[i, 'syn_strength']
        else:
            connection_matrix[post_index, pre_index] += downstream_connections.loc[i, 'syn_strength']
        connection_counts[post_index, pre_index] += 1
    # calculate the geometric mean of the connections
    # make zeros 1
    # connection_counts[connection_counts == 0] = 1
    # connection_matrix = np.power(connection_matrix, 1/connection_counts)
    # propagate the values
    output_values = np.dot(connection_matrix, input_values)
    # mixing strength (variance of the inputs received by each neuron)
    mixing_strength = np.std(connection_matrix*input_values, axis=1)

    return downstream_neurons, output_values, mixing_strength

# ==============================================================================
# Reduction by Cell Type: Helper Function
# ==============================================================================

def reduce_by_cell_type(full_matrix, index_slices):
    """
    Given a full connectivity matrix (neurons x neurons) and a list of index arrays
    (one per cell type), average each block (i.e. connectivity between two cell types)
    to produce a reduced connectivity matrix.
    """
    num_types = len(index_slices)
    reduced = np.zeros((num_types, num_types))
    for i, inds_i in enumerate(index_slices):
        for j, inds_j in enumerate(index_slices):
            block = full_matrix[np.ix_(inds_i, inds_j)]
            reduced[i, j] = block.mean()
    return reduced

#################################################
# Helper Functions for Graph Construction
#################################################
def create_graph_from_matrix(connectivity_matrix, nodes, threshold=0):
    """
    Given a connectivity matrix (assumed to be neurons x neurons) and a list of node labels,
    create a directed graph. Only edges with weight > threshold are added.
    """
    G = nx.DiGraph()
    # Add nodes
    for node in nodes:
        G.add_node(node)
    # Add edges: iterate over the matrix
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = connectivity_matrix[i, j]
            if weight > threshold:
                G.add_edge(nodes[i], nodes[j], weight=weight)
    return G

#################################################
# Community Detection using Louvain Method
#################################################
def detect_communities(G):
    """
    Detect communities using the Louvain method. Note that Louvain works on undirected graphs,
    so we convert the directed graph to undirected.
    """
    if G.is_directed():
        UG = G.to_undirected()
    else:
        UG = G
    # Compute the best partition. 'weight' is used for weighted networks.
    partition = community_louvain.best_partition(UG, weight='weight')
    return partition

#################################################
# Network Embedding using Node2Vec
#################################################
def perform_node2vec_embedding(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1):
    """
    Compute node embeddings using the node2vec algorithm.
    Returns a dictionary mapping node -> embedding vector.
    """
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, p=p, q=q, weight_key='weight', workers=4)
    model = node2vec.fit(window=10, min_count=1)
    embeddings = {node: model.wv[node] for node in G.nodes()}
    return embeddings

#################################################
# Optional: Visualization Helpers
#################################################
def plot_communities(G, partition, pos=None, node_size=50):
    """
    Plot the network with nodes colored by community.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_size,
                           cmap=cmap,
                           node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title("Community Detection (Louvain)")
    plt.show()

def plot_embedding(embeddings, partition, nodes, connectivity_matrix, threshold=0):
    """
    Reduce the high-dimensional embeddings to 2D using UMAP and plot them.
    """
    embedding_array = np.array([embeddings[node] for node in nodes])
    umap = UMAP(n_components=2, random_state=42)
    umap_embedding = umap.fit_transform(embedding_array)
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.viridis
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=list(partition.values()), cmap=cmap, s=50, zorder=10)
    # label the nodes
    for i, node in enumerate(nodes):
        plt.text(umap_embedding[i, 0], umap_embedding[i, 1], node, fontsize=8)
    # Draw edges for connections above threshold
    for i in range(connectivity_matrix.shape[0]):
        for j in range(connectivity_matrix.shape[1]):
            if connectivity_matrix[i, j] > threshold:
                plt.plot([umap_embedding[i, 0], umap_embedding[j, 0]],
                         [umap_embedding[i, 1], umap_embedding[j, 1]], '-', color='lightgray', alpha=0.3, linewidth=connectivity_matrix[i, j]/np.max(connectivity_matrix))
    plt.title("Node Embeddings (UMAP)")
    plt.show()

# =============================================================================  
# ==============================================================================
# Connectivity
# ==============================================================================
#%
import networkx as nx

# ==============================================================================
# First Order Connectivity: Full Method
# ==============================================================================
# Try Rishika's N-order analysis, up to 3

def create_first_order_connection_matrix(pre_neurons, post_neurons, connections, syn_var='syn_strength'):
    filtered_connections = connections[
        connections['pre_root_id'].isin(pre_neurons) & 
        connections['post_root_id'].isin(post_neurons)
    ]
    grouped = filtered_connections.groupby(['pre_root_id', 'post_root_id'])[syn_var].sum().reset_index()
    connection_matrix_df = grouped.pivot(index='pre_root_id', columns='post_root_id', values=syn_var).fillna(0)
    connection_matrix_df = connection_matrix_df.reindex(index=pre_neurons, columns=post_neurons, fill_value=0)
    
    return connection_matrix_df.values

# ==============================================================================
# Second Order Connectivity: Full Method via Intermediate Neurons
# ==============================================================================

def create_second_order_connection_matrix(pre_neurons, post_neurons, connections,
                                          return_intermediates=False, syn_var='syn_strength'):
    # Filter connections for valid synapse strengths
    valid_connections = connections.copy()
    valid_connections = valid_connections[~np.isnan(valid_connections[syn_var])]
    #valid_connections = valid_connections[(valid_connections[syn_var] > 0) & (valid_connections[syn_var] < 1)]
    
    # Determine downstream (from pre_neurons) and upstream (to post_neurons) neurons
    downstream = valid_connections[valid_connections['pre_root_id'].isin(pre_neurons)]
    downstream_neurons = downstream['post_root_id'].unique()
    
    upstream = valid_connections[valid_connections['post_root_id'].isin(post_neurons)]
    upstream_neurons = upstream['pre_root_id'].unique()
    
    # Common intermediate neurons between the two groups
    common_neurons = np.intersect1d(downstream_neurons, upstream_neurons)

    # remove any neurons in common_neurons that are also in pre_neurons or post_neurons to prevent cycles
    common_neurons = common_neurons[~np.isin(common_neurons, np.concatenate([pre_neurons, post_neurons]))]
    
    # Build pre->common matrix
    pre_common_connections = connections[
        connections['pre_root_id'].isin(pre_neurons) & 
        connections['post_root_id'].isin(common_neurons)
    ]
    pre_common_group = pre_common_connections.groupby(['pre_root_id', 'post_root_id'])[syn_var].sum().reset_index()
    pre_common_df = pre_common_group.pivot(index='pre_root_id', columns='post_root_id', values=syn_var).fillna(0)
    pre_common_df = pre_common_df.reindex(index=pre_neurons, columns=common_neurons, fill_value=0)
    pre_common_matrix = pre_common_df.values

    # Build common->post matrix
    common_post_connections = connections[
        connections['pre_root_id'].isin(common_neurons) &
        connections['post_root_id'].isin(post_neurons)
    ]
    common_post_group = common_post_connections.groupby(['pre_root_id', 'post_root_id'])[syn_var].sum().reset_index()
    common_post_df = common_post_group.pivot(index='pre_root_id', columns='post_root_id', values=syn_var).fillna(0)
    common_post_df = common_post_df.reindex(index=common_neurons, columns=post_neurons, fill_value=0)
    common_post_matrix = common_post_df.values

    # Multiply the matrices to get second order connectivity and take geometric mean
    second_order = np.dot(pre_common_matrix, common_post_matrix)
    second_order = np.power(second_order, 1/2)
   
    if return_intermediates:
        return second_order, common_neurons, [pre_common_matrix, common_post_matrix] 
    
    return second_order, common_neurons


# ==============================================================================
# Third Order Connectivity: Full Method via Intermediate Neurons
# ==============================================================================

def create_third_order_connection_matrix(pre_neurons, post_neurons, connections,
                                          return_intermediates=False):
    """
    Compute a third order connection matrix between pre_neurons and post_neurons.
    Here a third order connection is defined as a path:
      pre_neuron -> intermediate1 -> intermediate2 -> post_neuron,
    where intermediate1 and intermediate2 are chosen by:
      - I1: all neurons downstream of pre_neurons
      - I2: all neurons upstream of post_neurons
    Only direct connections are considered at each step.
    
    The final third order connectivity is computed as the product of three matrices,
    with a cube root taken to (optionally) keep values in a comparable range.
    """
    # Filter valid connections based on synapse strength
    valid_connections = connections.copy()
    valid_connections = valid_connections[~np.isnan(valid_connections['syn_strength'])]
    valid_connections = valid_connections[(valid_connections['syn_strength'] > 0) & 
                                          (valid_connections['syn_strength'] < 1)]
    
    # Determine intermediate sets:
    # I1: neurons that appear as targets from pre_neurons
    pre_downstream = valid_connections[valid_connections['pre_root_id'].isin(pre_neurons)]
    I1 = np.unique(pre_downstream['post_root_id'])
    # remove any neurons in I1 that are also in pre_neurons or post_neurons to prevent cycles
    I1 = I1[~np.isin(I1, np.concatenate([pre_neurons, post_neurons]))]

    # I2: neurons that appear as sources to post_neurons
    post_upstream = valid_connections[valid_connections['post_root_id'].isin(post_neurons)]
    I2 = np.unique(post_upstream['pre_root_id'])
    # remove any neurons in I2 that are also in pre_neurons or post_neurons to prevent cycles
    I2 = I2[~np.isin(I2, np.concatenate([pre_neurons, post_neurons]))]

    # remove any neurons in I2 that are also in I1 to prevent cycles
    I2 = I2[~np.isin(I2, I1)] # I2: neurons that appear as sources to post_neurons

    
    # --- Build pre -> I1 matrix ---
    pre_I1 = valid_connections[
        valid_connections['pre_root_id'].isin(pre_neurons) &
        valid_connections['post_root_id'].isin(I1)
    ]
    pre_I1_group = pre_I1.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    pre_I1_df = pre_I1_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    pre_I1_df = pre_I1_df.reindex(index=pre_neurons, columns=I1, fill_value=0)
    pre_I1_matrix = pre_I1_df.values

    # --- Build I1 -> I2 matrix ---
    I1_I2 = valid_connections[
        valid_connections['pre_root_id'].isin(I1) &
        valid_connections['post_root_id'].isin(I2)
    ]
    I1_I2_group = I1_I2.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    I1_I2_df = I1_I2_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    I1_I2_df = I1_I2_df.reindex(index=I1, columns=I2, fill_value=0)
    I1_I2_matrix = I1_I2_df.values

    # --- Build I2 -> post matrix ---
    I2_post = valid_connections[
        valid_connections['pre_root_id'].isin(I2) &
        valid_connections['post_root_id'].isin(post_neurons)
    ]
    I2_post_group = I2_post.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    I2_post_df = I2_post_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    I2_post_df = I2_post_df.reindex(index=I2, columns=post_neurons, fill_value=0)
    I2_post_matrix = I2_post_df.values

    # --- Multiply the three matrices ---
    third_order = np.dot(np.dot(pre_I1_matrix, I1_I2_matrix), I2_post_matrix)
    # Apply the cube root (geometric mean) to temper large values
    third_order = np.power(third_order, 1/3)
   
    if return_intermediates:
        return third_order, I1, I2, [pre_I1_matrix, I1_I2_matrix, I2_post_matrix]
     
    return third_order, I1, I2


# ==============================================================================
# FOURTH Order Connectivity: Full Method via Intermediate Neurons
# ==============================================================================

def create_fourth_order_connection_matrix(pre_neurons, post_neurons, connections,
                                          return_intermediates=False):
    """
    Modeled after Rishika's N-order analysis, up to 4.
    Compute a fourth order connection matrix between pre_neurons and post_neurons.
    Here a fourth order connection is defined as a path:
      pre_neuron -> intermediate1 -> intermediate2 -> intermediate3 -> post_neuron,
    where intermediate1 and intermediate2 are chosen by:
      - I1: all neurons downstream of pre_neurons
      - I2: all neurons downstream of I1 neurons and upstream of I3 neurons 
      - I3: all neurons upsream of POST neurons
    Only direct connections are considered at each step.
    
    The final third order connectivity is computed as the product of three matrices,
    with a cube root taken to (optionally) keep values in a comparable range.
    """
    # Filter valid connections based on synapse strength
    valid_connections = connections.copy()
    valid_connections = valid_connections[~np.isnan(valid_connections['syn_strength'])]
    valid_connections = valid_connections[(valid_connections['syn_strength'] > 0) & 
                                          (valid_connections['syn_strength'] < 1)]
    
    # Determine intermediate sets:
    # I1: neurons that appear as targets from pre_neurons
    pre_downstream = valid_connections[valid_connections['pre_root_id'].isin(pre_neurons)]
    I1 = np.unique(pre_downstream['post_root_id'])
    # remove any neurons in I1 that are also in pre_neurons or post_neurons to prevent cycles
    I1 = I1[~np.isin(I1, np.concatenate([pre_neurons, post_neurons]))]

    # I2:  neurons that appear as targets from I1 neurons (and sources to I3 neurons)
    pre_I1  = valid_connections[valid_connections['pre_root_id'].isin(I1)] 
    I2 = np.unique(pre_I1['post_root_id'])
    # remove any neurons in I1 that are also in pre_neurons or post_neurons to prevent cycles
    I2 = I2[~np.isin(I2, np.concatenate([pre_neurons, post_neurons]))]
    # remove any neurons in I2 that are also in I1 to prevent cycles
    I2 = I2[~np.isin(I2, I1)] # I2: neurons that appear as sources to I3 neurons

    # I3: neurons that appear as sources to post_neurons
    post_upstream = valid_connections[valid_connections['post_root_id'].isin(post_neurons)]
    I3 = np.unique(post_upstream['pre_root_id'])
    # remove any neurons in I2 that are also in pre_neurons or post_neurons to prevent cycles
    I3 = I3[~np.isin(I3, np.concatenate([pre_neurons, post_neurons]))]

    # remove any neurons in I2 that are also in I1 to prevent cycles
    I3 = I3[~np.isin(I3, I2)] # I3: neurons that appear as sources to post_neurons

    
    # --- Build pre -> I1 matrix ---
    pre_I1 = valid_connections[
        valid_connections['pre_root_id'].isin(pre_neurons) &
        valid_connections['post_root_id'].isin(I1)
    ]
    pre_I1_group = pre_I1.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    pre_I1_df = pre_I1_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    pre_I1_df = pre_I1_df.reindex(index=pre_neurons, columns=I1, fill_value=0)
    pre_I1_matrix = pre_I1_df.values

    # --- Build I1 -> I2 matrix ---
    I1_I2 = valid_connections[
        valid_connections['pre_root_id'].isin(I1) &
        valid_connections['post_root_id'].isin(I2)
    ]
    I1_I2_group = I1_I2.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    I1_I2_df = I1_I2_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    I1_I2_df = I1_I2_df.reindex(index=I1, columns=I2, fill_value=0)
    I1_I2_matrix = I1_I2_df.values

    # --- Build I2 -> I3 matrix ---
    I2_I3 = valid_connections[
        valid_connections['pre_root_id'].isin(I2) &
        valid_connections['post_root_id'].isin(I3)
    ]
    I2_I3_group = I2_I3.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    I2_I3_df = I2_I3_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    I2_I3_df = I2_I3_df.reindex(index=I2, columns=I3, fill_value=0)
    I2_I3_matrix = I2_I3_df.values

    # --- Build I3 -> post matrix ---
    I3_post = valid_connections[
        valid_connections['pre_root_id'].isin(I3) &
        valid_connections['post_root_id'].isin(post_neurons)
    ]
    I3_post_group = I3_post.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    I3_post_df = I3_post_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    I3_post_df = I3_post_df.reindex(index=I3, columns=post_neurons, fill_value=0)
    I3_post_matrix = I3_post_df.values

    # --- Multiply the three matrices ---
    fourth_order = np.dot(np.dot(np.dot(pre_I1_matrix, I1_I2_matrix), I2_I3_matrix),  I3_post_matrix)
    # Apply the cube root (geometric mean) to temper large values
    fourth_order = np.power(fourth_order, 1/4)
   
    if return_intermediates:
        return fourth_order, I1, I2, I3, [pre_I1_matrix, I1_I2_matrix, I2_I3_matrix, I3_post_matrix]
    
    return fourth_order, I1, I2, I3

#%% Path reconstruction

from scipy.sparse import csr_matrix, find

# ============================================================================
# 1st Order: pre -> post
# ============================================================================
def get_explicit_paths_sparse_first(pre_neurons, post_neurons, pre_post_matrix):
    """
    pre_post_matrix: dense array or matrix with shape (n_pre, n_post)
    pre_neurons: list or array of length n_pre mapping rows to neuron IDs.
    post_neurons: list or array of length n_post mapping columns to neuron IDs.
    
    Returns a DataFrame with columns ["pre", "post"].
    """
    sp = csr_matrix(pre_post_matrix)
    rows, cols, _ = find(sp)
    df = pd.DataFrame({
        "pre_idx": rows,
        "post_idx": cols
    })
    df["pre"] = df["pre_idx"].apply(lambda i: pre_neurons[i])
    df["post"] = df["post_idx"].apply(lambda i: post_neurons[i])
    return df[["pre", "post"]]


# ============================================================================
# 2nd Order: pre -> I1 -> post
# ============================================================================
def get_explicit_paths_sparse_second(pre_neurons, I1_neurons, post_neurons,
                                     pre_I1_matrix, I1_post_matrix):
    """
    pre_I1_matrix: dense array of shape (n_pre, n_I1)
    I1_post_matrix: dense array of shape (n_I1, n_post)
    
    Returns a DataFrame with columns ["pre", "I1", "post"].
    """
    # Convert to sparse
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_post_matrix)
    
    # Get nonzero indices for pre->I1
    rows_A, cols_A, _ = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    # Get nonzero indices for I1->post
    rows_B, cols_B, _ = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "post_idx": cols_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["post"] = df_B["post_idx"].apply(lambda i: post_neurons[i])
    
    # Merge on the intermediate neuron (I1)
    merge_df = pd.merge(df_A, df_B, on="I1", how="inner")
    return merge_df[["pre", "I1", "post"]]


# ============================================================================
# 3rd Order: pre -> I1 -> I2 -> post
# ============================================================================
def get_explicit_paths_sparse_third(pre_neurons, I1_neurons, I2_neurons, post_neurons,
                                    pre_I1_matrix, I1_I2_matrix, I2_post_matrix):
    """
    pre_I1_matrix: dense array of shape (n_pre, n_I1)
    I1_I2_matrix: dense array of shape (n_I1, n_I2)
    I2_post_matrix: dense array of shape (n_I2, n_post)
    
    Returns a DataFrame with columns ["pre", "I1", "I2", "post"].
    """
    # Convert to sparse
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_I2_matrix)
    sp_C = csr_matrix(I2_post_matrix)
    
    # Pre -> I1
    rows_A, cols_A, _ = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    # I1 -> I2
    rows_B, cols_B, _ = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "I2_idx": cols_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["I2"] = df_B["I2_idx"].apply(lambda i: I2_neurons[i])
    
    # I2 -> post
    rows_C, cols_C, _ = find(sp_C)
    df_C = pd.DataFrame({
        "I2_idx": rows_C,
        "post_idx": cols_C
    })
    df_C["I2"] = df_C["I2_idx"].apply(lambda i: I2_neurons[i])
    df_C["post"] = df_C["post_idx"].apply(lambda i: post_neurons[i])
    
    # Merge DataFrames
    merge_AB = pd.merge(df_A, df_B, on="I1", how="inner")
    merge_ABC = pd.merge(merge_AB, df_C, on="I2", how="inner")
    
    return merge_ABC[["pre", "I1", "I2", "post"]]


# ============================================================================
# 4th Order: pre -> I1 -> I2 -> I3 -> post
# ============================================================================
def get_explicit_paths_sparse_fourth(pre_neurons, I1_neurons, I2_neurons, I3_neurons, post_neurons,
                                     pre_I1_matrix, I1_I2_matrix, I2_I3_matrix, I3_post_matrix):
    """
    pre_I1_matrix: dense array of shape (n_pre, n_I1)
    I1_I2_matrix: dense array of shape (n_I1, n_I2)
    I2_I3_matrix: dense array of shape (n_I2, n_I3)
    I3_post_matrix: dense array of shape (n_I3, n_post)
    
    Returns a DataFrame with columns ["pre", "I1", "I2", "I3", "post"].
    """
    # Convert to sparse
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_I2_matrix)
    sp_C = csr_matrix(I2_I3_matrix)
    sp_D = csr_matrix(I3_post_matrix)
    
    # Pre -> I1
    rows_A, cols_A, _ = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    # I1 -> I2
    rows_B, cols_B, _ = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "I2_idx": cols_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["I2"] = df_B["I2_idx"].apply(lambda i: I2_neurons[i])
    
    # I2 -> I3
    rows_C, cols_C, _ = find(sp_C)
    df_C = pd.DataFrame({
        "I2_idx": rows_C,
        "I3_idx": cols_C
    })
    df_C["I2"] = df_C["I2_idx"].apply(lambda i: I2_neurons[i])
    df_C["I3"] = df_C["I3_idx"].apply(lambda i: I3_neurons[i])
    
    # I3 -> post
    rows_D, cols_D, _ = find(sp_D)
    df_D = pd.DataFrame({
        "I3_idx": rows_D,
        "post_idx": cols_D
    })
    df_D["I3"] = df_D["I3_idx"].apply(lambda i: I3_neurons[i])
    df_D["post"] = df_D["post_idx"].apply(lambda i: post_neurons[i])
    
    # Merge DataFrames step by step
    merge_AB = pd.merge(df_A, df_B, on="I1", how="inner")
    merge_ABC = pd.merge(merge_AB, df_C, on="I2", how="inner")
    merge_ABCD = pd.merge(merge_ABC, df_D, on="I3", how="inner")
    
    return merge_ABCD[["pre", "I1", "I2", "I3", "post"]]

def validate_path(path, connections):
    """
    Validate a path by checking if each connection in the path exists in the connections DataFrame.
    
    Parameters:
      path (list): List of tuples representing a path.
      connections (DataFrame): DataFrame of synapse connections.
      
    Returns:
      bool: True if all connections in the path exist in the connections DataFrame.
      
    # Example usage:
    path is a row of the DataFrame returned by get_explicit_paths_sparse_fourth:
        pre_id     720575940612457139
        I1_id      720575940597856265
        I2_id      720575940606161586
        I3_id      720575940606114220
        post_id    720575940615833574
        
    validate_path(path, incl_connections) returns TRUE if all connections exist
    """
    for i in range(len(path)-1):
        pre, post = path[i], path[i+1]
        if len(connections[(connections['pre_root_id']==pre) & (connections['post_root_id']==post)]) == 0:
            return False
    return True

#%% Include strength
# ============================================================================
# 1st Order: pre -> post, with strength (geometric mean == product since only one edge)
# ============================================================================
def get_explicit_paths_sparse_first_with_strength(pre_neurons, post_neurons, pre_post_matrix, agg_method="product"):
    sp = csr_matrix(pre_post_matrix)
    rows, cols, data = find(sp)
    df = pd.DataFrame({
        "pre_idx": rows,
        "post_idx": cols,
        "w": data
    })
    df["pre"] = df["pre_idx"].apply(lambda i: pre_neurons[i])
    df["post"] = df["post_idx"].apply(lambda i: post_neurons[i])
    # For a single connection, product or geometric mean are identical.
    df["strength"] = df["w"]
    return df[["pre", "post", "strength"]]


# ============================================================================
# 2nd Order: pre -> I1 -> post, with strength
# ============================================================================
def get_explicit_paths_sparse_second_with_strength(pre_neurons, I1_neurons, post_neurons,
                                                   pre_I1_matrix, I1_post_matrix, agg_method="product"):
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_post_matrix)
    
    rows_A, cols_A, data_A = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A,
        "w_A": data_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    rows_B, cols_B, data_B = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "post_idx": cols_B,
        "w_B": data_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["post"] = df_B["post_idx"].apply(lambda i: post_neurons[i])
    
    merge_df = pd.merge(df_A, df_B, on="I1", how="inner")
    product_strength = merge_df["w_A"] * merge_df["w_B"]
    # Use geometric mean if requested
    if agg_method == "geom":
        merge_df["strength"] = product_strength ** (1/2)
    else:
        merge_df["strength"] = product_strength
    return merge_df[["pre", "I1", "post", "strength"]]


# ============================================================================
# 3rd Order: pre -> I1 -> I2 -> post, with strength
# ============================================================================
def get_explicit_paths_sparse_third_with_strength(pre_neurons, I1_neurons, I2_neurons, post_neurons,
                                                  pre_I1_matrix, I1_I2_matrix, I2_post_matrix, agg_method="product"):
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_I2_matrix)
    sp_C = csr_matrix(I2_post_matrix)
    
    # Pre -> I1
    rows_A, cols_A, data_A = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A,
        "w_A": data_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    # I1 -> I2
    rows_B, cols_B, data_B = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "I2_idx": cols_B,
        "w_B": data_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["I2"] = df_B["I2_idx"].apply(lambda i: I2_neurons[i])
    
    # I2 -> post
    rows_C, cols_C, data_C = find(sp_C)
    df_C = pd.DataFrame({
        "I2_idx": rows_C,
        "post_idx": cols_C,
        "w_C": data_C
    })
    df_C["I2"] = df_C["I2_idx"].apply(lambda i: I2_neurons[i])
    df_C["post"] = df_C["post_idx"].apply(lambda i: post_neurons[i])
    
    merge_AB = pd.merge(df_A, df_B, on="I1", how="inner")
    merge_ABC = pd.merge(merge_AB, df_C, on="I2", how="inner")
    
    product_strength = merge_ABC["w_A"] * merge_ABC["w_B"] * merge_ABC["w_C"]
    if agg_method == "geom":
        merge_ABC["strength"] = product_strength ** (1/3)
    else:
        merge_ABC["strength"] = product_strength
    return merge_ABC[["pre", "I1", "I2", "post", "strength"]]


# ============================================================================
# 4th Order: pre -> I1 -> I2 -> I3 -> post, with strength
# ============================================================================
def get_explicit_paths_sparse_fourth_with_strength(pre_neurons, I1_neurons, I2_neurons, I3_neurons, post_neurons,
                                                   pre_I1_matrix, I1_I2_matrix, I2_I3_matrix, I3_post_matrix, agg_method="product"):
    sp_A = csr_matrix(pre_I1_matrix)
    sp_B = csr_matrix(I1_I2_matrix)
    sp_C = csr_matrix(I2_I3_matrix)
    sp_D = csr_matrix(I3_post_matrix)
    
    # Pre -> I1
    rows_A, cols_A, data_A = find(sp_A)
    df_A = pd.DataFrame({
        "pre_idx": rows_A,
        "I1_idx": cols_A,
        "w_A": data_A
    })
    df_A["pre"] = df_A["pre_idx"].apply(lambda i: pre_neurons[i])
    df_A["I1"] = df_A["I1_idx"].apply(lambda i: I1_neurons[i])
    
    # I1 -> I2
    rows_B, cols_B, data_B = find(sp_B)
    df_B = pd.DataFrame({
        "I1_idx": rows_B,
        "I2_idx": cols_B,
        "w_B": data_B
    })
    df_B["I1"] = df_B["I1_idx"].apply(lambda i: I1_neurons[i])
    df_B["I2"] = df_B["I2_idx"].apply(lambda i: I2_neurons[i])
    
    # I2 -> I3
    rows_C, cols_C, data_C = find(sp_C)
    df_C = pd.DataFrame({
        "I2_idx": rows_C,
        "I3_idx": cols_C,
        "w_C": data_C
    })
    df_C["I2"] = df_C["I2_idx"].apply(lambda i: I2_neurons[i])
    df_C["I3"] = df_C["I3_idx"].apply(lambda i: I3_neurons[i])
    
    # I3 -> post
    rows_D, cols_D, data_D = find(sp_D)
    df_D = pd.DataFrame({
        "I3_idx": rows_D,
        "post_idx": cols_D,
        "w_D": data_D
    })
    df_D["I3"] = df_D["I3_idx"].apply(lambda i: I3_neurons[i])
    df_D["post"] = df_D["post_idx"].apply(lambda i: post_neurons[i])
    
    merge_AB = pd.merge(df_A, df_B, on="I1", how="inner")
    merge_ABC = pd.merge(merge_AB, df_C, on="I2", how="inner")
    merge_ABCD = pd.merge(merge_ABC, df_D, on="I3", how="inner")
    
    product_strength = merge_ABCD["w_A"] * merge_ABCD["w_B"] * merge_ABCD["w_C"] * merge_ABCD["w_D"]
    if agg_method == "geom":
        merge_ABCD["strength"] = product_strength ** (1/4)
    else:
        merge_ABCD["strength"] = product_strength
    return merge_ABCD[["pre", "I1", "I2", "I3", "post", "strength"]]


def standardize_paths_df(df, standard_columns=['pre', 'I1', 'I2', 'I3', 'post', 'strength']):
    # For each column in the standard list, add it if missing
    for col in standard_columns:
        if col not in df.columns:
            df[col] = -1 #np.nan
    # Order the columns
    return df[standard_columns]

#%%

import pickle as pkl
import os
import pandas as pd


def get_first_order(pre_neurons, post_neurons, incl_connections, 
                    matrix_file=None, recalculate=False, syn_var='syn_strength'):
    '''
    Calculate 1st order connections between pre_neurons and post_neurons.
    Save or load from matrix_file. Wrapper for creating matrix and getting paths.
    '''
    if not os.path.exists(matrix_file) or recalculate:
        # Create a connection matrix for DANs and MBONs using the vectorized approach
        VC_LC10a_connection_matrix_first = create_first_order_connection_matrix(pre_neurons, post_neurons,
                                                                        incl_connections, syn_var=syn_var)
        #np.save(matrix_file, VC_LC10a_connection_matrix_first)

        #[pre_common_matrix, common_post_matrix] 
        #pre_I1_matrix_second, I1_post_matrix_second = intermediate_mats_second
        print("1st order connections between {} VCs and {} LCs".format(len(pre_neurons), len(post_neurons)))
        print(VC_LC10a_connection_matrix_first.shape) 
        
        # Get start time
        #start_time = time.time()
        paths_first = get_explicit_paths_sparse_first_with_strength(
                                            pre_neurons, 
                                            post_neurons, 
                                            VC_LC10a_connection_matrix_first)
        # Print time elapsed in sec.
        #print("--- %s seconds ---" % (time.time() - start_time))
        # Save all the outputs calculated for fourth order connections
        with open(matrix_file, 'wb') as f:
            pkl.dump((VC_LC10a_connection_matrix_first, paths_first), f)
        print("Saved: \n{}".format(matrix_file)) 
    else:
        with open(matrix_file, 'rb') as f:
            VC_LC10a_connection_matrix_first, paths_first = pkl.load(f)
        #VC_LC10a_connection_matrix_first = np.load(matrix_file)
    return VC_LC10a_connection_matrix_first, paths_first

def get_second_order(pre_neurons, post_neurons, incl_connections, 
                     matrix_file=None, recalculate=False, return_intermediates=False,
                     syn_var='syn_strength'):
    if not os.path.exists(matrix_file) or recalculate:
        # Create a connection matrix for DANs and MBONs using the vectorized approach
        VC_LC10a_connection_matrix_second, common_neurons_second, intermediate_mats_second = create_second_order_connection_matrix(
                                                                                        pre_neurons, post_neurons,
                                                                                        incl_connections, return_intermediates=True,
                                                                                        syn_var=syn_var)
        #np.save(matrix_file, VC_LC10a_connection_matrix_second)

        #[pre_common_matrix, common_post_matrix] 
        pre_I1_matrix_second, I1_post_matrix_second = intermediate_mats_second
        # note: explicit_paths gets strengths by the pre_I1_matrix_second, etc. matrices 
        # so syn counts (min NON-zero should be 10)
        print("2nd order connections between {} VCs and {} LCs".format(len(pre_neurons), len(post_neurons)))
        for i, m in enumerate(intermediate_mats_second):
            print(i, m.shape) 
        
        # Get start time
        #start_time = time.time()
        paths_second = get_explicit_paths_sparse_second_with_strength(
                                            pre_neurons,            # array of pre neuron IDs
                                            common_neurons_second,         # array of neurons between pre and post 
                                            post_neurons,           # array of post neuron IDs 
                                            pre_I1_matrix_second,   # dense matrix: shape (n_pre, n_common) 
                                            I1_post_matrix_second   # dense matrix: shape (n_common, n_post)
                                            )
        # Print time elapsed in sec.
        #print("--- %s seconds ---" % (time.time() - start_time))
        # Save all the outputs calculated for fourth order connections
        with open(matrix_file, 'wb') as f:
            pkl.dump((VC_LC10a_connection_matrix_second, common_neurons_second, 
                    intermediate_mats_second, paths_second), f)
        print("Saved: \n{}".format(matrix_file)) 
    else:
        with open(matrix_file, 'rb') as f:
            VC_LC10a_connection_matrix_second, common_neurons_second, intermediate_mats_second, paths_second = pkl.load(f)
        pre_I1_matrix_second, I1_post_matrix_second = intermediate_mats_second
        
    if return_intermediates:
        return VC_LC10a_connection_matrix_second, paths_second, (common_neurons_second, pre_I1_matrix_second, I1_post_matrix_second)
    else: 
        return VC_LC10a_connection_matrix_second, paths_second


def get_third_order(pre_neurons, post_neurons, incl_connections, 
                    matrix_file=None, recalculate=False, return_intermediates=False):
    if not os.path.exists(matrix_file) or recalculate:
        # Create a connection matrix for DANs and MBONs using the vectorized approach
        VC_LC10a_connection_matrix_third, I1_neurons_third, I2_neurons_third, intermediate_mats_third = create_third_order_connection_matrix(pre_neurons, post_neurons,
                                                                                        incl_connections, return_intermediates=True)
    #    with open(matrix_file, 'wb') as f:
    #        pkl.dump((VC_LC10a_connection_matrix_third, I1, I2, intermediates_third), f)    
        #np.save(matrix_file, VC_LC10a_connection_matrix_third)
        # Get intermediates

        #[pre_I1_matrix, I1_I2_matrix, I2_post_matrix]
        pre_I1_matrix_third, I1_I2_matrix_third, I2_post_matrix_third = intermediate_mats_third
        print("3rd order connections between {} VCs and {} LCs".format(len(pre_neurons), len(post_neurons)))
        for i, m in enumerate(intermediate_mats_third):
            print(i, m.shape) 
        #%
        # Get start time
        #start_time = time.time()
        paths_third = get_explicit_paths_sparse_third_with_strength(
                                            pre_neurons,            # e.g., array of pre_neuron IDs
                                            I1_neurons_third,       # array of I1 neuron IDs
                                            I2_neurons_third,       # array of I2 neuron IDs
                                            post_neurons,           # array of post neuron IDs
                                            pre_I1_matrix_third,    # dense matrix: shape (n_pre, n_I1)
                                            I1_I2_matrix_third,     # dense matrix: shape (n_I1, n_I2)
                                            I2_post_matrix_third    # dense matrix: shape (n_I2, n_post)
                                    )
        # Print time elapsed in sec.
        #print("--- %s seconds ---" % (time.time() - start_time))
        # Save all the outputs calculated for fourth order connections
        with open(matrix_file, 'wb') as f:
            pkl.dump((VC_LC10a_connection_matrix_third, I1_neurons_third, I2_neurons_third, 
                    intermediate_mats_third, paths_third), f)
        print("Saved: \n{}".format(matrix_file)) 

    else:
        with open(matrix_file, 'rb') as f:
            VC_LC10a_connection_matrix_third, I1_neurons_third, I2_neurons_third, intermediate_mats_third, paths_third = pkl.load(f)
        # unpack intermediate matrices
        pre_I1_matrix_third, I1_I2_matrix_third, I2_post_matrix_third = intermediate_mats_third
         
        #VC_LC10a_connection_matrix_third = np.load(matrix_file)
    
    if return_intermediates:
        return VC_LC10a_connection_matrix_third, paths_third, (I1_neurons_third, I2_neurons_third, pre_I1_matrix_third, I1_I2_matrix_third, I2_post_matrix_third)    

    return VC_LC10a_connection_matrix_third, paths_third


def get_fourth_order(pre_neurons, post_neurons, incl_connections, 
                     matrix_file=None, recalculate=False, 
                     return_intermediates=False):
    if not os.path.exists(matrix_file) or recalculate:
        # Create a connection matrix for DANs and MBONs using the vectorized approach
        VC_LC10a_connection_matrix_fourth, I1_neurons_fourth, I2_neurons_fourth, I3_neurons_fourth, intermediate_mats_fourth = create_fourth_order_connection_matrix(
                                                                                                pre_neurons, post_neurons,
                                                                                                incl_connections,
                                                                                                return_intermediates=True)
        #%  
        pre_I1_matrix_fourth, I1_I2_matrix_fourth, I2_I3_matrix_fourth, I3_post_matrix_fourth = intermediate_mats_fourth
        print("4th order connections between {} VCs and {} LCs".format(len(pre_neurons), len(post_neurons)))
        #for i, m in enumerate(intermediate_mats):
        #    print(i, m.shape) 
        #%
        # Get start time
        #start_time = time.time()
        paths_fourth = get_explicit_paths_sparse_fourth_with_strength(
                                            pre_neurons, 
                                            I1_neurons_fourth,      # array of I1 IDs (pre -> I1)
                                            I2_neurons_fourth,      # array of I2 IDs (I1 -> I2) 
                                            I3_neurons_fourth,      # array of I3 IDs (I2 -> I3) 
                                            post_neurons,           # array of post IDs (I3 -> post)
                                            pre_I1_matrix_fourth,   # dense array: (n_pre, n_I1)
                                            I1_I2_matrix_fourth,    # dense array: (n_I1, n_I2)
                                            I2_I3_matrix_fourth,    # dense array: (n_I2, n_I3)
                                            I3_post_matrix_fourth   # dense array: (n_I3, n_post)
                                            )

        #paths_fourth = get_explicit_paths(pre_neurons, I1_neurons, I2_neurons, I3_neurons, post_neurons,
        #                                    pre_I1_matrix, I1_I2_matrix, I2_I3_matrix, I3_post_matrix)
        # Print time elapsed in sec.
        #print("--- %s seconds ---" % (time.time() - start_time))
        # Save all the outputs calculated for fourth order connections
        with open(matrix_file, 'wb') as f:
            pkl.dump((VC_LC10a_connection_matrix_fourth, I1_neurons_fourth, I2_neurons_fourth, I3_neurons_fourth, intermediate_mats_fourth, paths_fourth), f)
        print("Saved: \n{}".format(matrix_file)) 
        #np.save(matrix_file, VC_LC10a_connection_matrix_fourth)
    else:
        #VC_LC10a_connection_matrix_fourth = np.load(matrix_file)
        with open(matrix_file, 'rb') as f:
            VC_LC10a_connection_matrix_fourth, I1_neurons_fourth, I2_neurons_fourth, I3_neurons_fourth, intermediate_mats_fourth, paths_fourth = pkl.load(f)
        # unpack intermediates 
        pre_I1_matrix_fourth, I1_I2_matrix_fourth, I2_I3_matrix_fourth, I3_post_matrix_fourth = intermediate_mats_fourth
        
    if return_intermediates:
        return VC_LC10a_connection_matrix_fourth, paths_fourth, (I1_neurons_fourth, I2_neurons_fourth, I3_neurons_fourth, pre_I1_matrix_fourth, I1_I2_matrix_fourth, I2_I3_matrix_fourth, I3_post_matrix_fourth)
    
    return VC_LC10a_connection_matrix_fourth, paths_fourth            

#%%
def combine_paths( paths_list, 
                  use_geom=True, standardize=True, recalculate=False): 
    '''
    
    Args:
    -----
    paths_list (list): [paths_first, paths_second, paths_third, paths_fourth]
    use_geom (bool): use geometric mean instead of product
    standardize (bool): standardize each order dataframe to combine into 1
    recalculate (bool): recalculate combined paths
   
    Returns:
    --------
    combined_paths_df (DataFrame): combined paths dataframe
     
    ''' 
    # GEOM MEAN instead of product?
    if use_geom:
        for pi, paths in enumerate(paths_list):
            paths['strength'] = paths['strength'] ** (1/(pi+1))
            # upate paths_list
            paths_list[pi] = paths
            
#         paths_first['strength'] = paths_first['strength'] ** (1/1)
#         paths_second['strength'] = paths_second['strength'] ** (1/2)
#         paths_third['strength'] = paths_third['strength'] ** (1/3)
#         paths_fourth['strength'] = paths_fourth['strength'] ** (1/4)
 
    # Define the standard columns you want:
    standard_columns = ["pre", "I1", "I2", "I3", "post", "strength"]

    # Standardize each dataframe:
    if standardize:
        for pi, paths in enumerate(paths_list):
            paths_list[pi] = standardize_paths_df(paths, standard_columns)
 
#     paths_first_std  = standardize_paths_df(paths_first.copy(), standard_columns)
#     paths_second_std = standardize_paths_df(paths_second.copy(), standard_columns)
#     paths_third_std  = standardize_paths_df(paths_third.copy(), standard_columns)
#     paths_fourth_std = standardize_paths_df(paths_fourth.copy(), standard_columns)
 
    # Concatenate them all into one dataframe:
    combined_paths_df = pd.concat(paths_list, ignore_index=True)
    #%

    return combined_paths_df
          