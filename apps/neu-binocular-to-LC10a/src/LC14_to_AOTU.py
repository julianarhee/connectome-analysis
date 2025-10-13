#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : LC14_to_AOTU.py
Created        : 2025/10/11 13:54:10
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/neu-binocular-to-LC10a/src
Author         : jyr
Last Modified  : 
'''
#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neuprint import Client
import neuprint as neu
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import neuprint_funcs as npfuncs

# %%
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

# %%
c = Client('neuprint.janelia.org', dataset='male-cns:v0.9', token=token)
c.fetch_version()

# %%

def plot_connectivity_matrix(conn_matrix, src_df, target_df,
                             src_L, src_R, tgt_L, tgt_R,
                             src_name='Source', target_name='Target',
                             use_instance=True, x_rotation=45, y_interval=5, x_interval=1,
                             threshold=None, cluster=False, figsize=(14, 12), cmap='viridis'):
    """Plot connectivity matrix with L/R grouping and optional threshold masking."""
    # Cluster if requested
    if cluster:
        print("Clustering matrix by cosine similarity...")
        conn_matrix, _, _ = npfuncs.cluster_connectivity_matrix(conn_matrix)
        # Update L/R counts after filtering (clustering removes neurons with <1 connections)
        src_L, src_R = npfuncs.split_ids_by_side(src_df)
        tgt_L, tgt_R = npfuncs.split_ids_by_side(target_df)
        # Filter to only include neurons that remain in clustered matrix
        src_L = [id for id in src_L if id in conn_matrix.index]
        src_R = [id for id in src_R if id in conn_matrix.index]
        tgt_L = [id for id in tgt_L if id in conn_matrix.columns]
        tgt_R = [id for id in tgt_R if id in conn_matrix.columns]
    
    # Get labels
    def get_labels(df, ids, use_instance):
        mapper = dict(zip(df['bodyId'], df['instance'])) if use_instance else {}
        return [mapper.get(i, str(i)) for i in ids]
    
    src_labels = get_labels(src_df, conn_matrix.index, use_instance)
    tgt_labels = get_labels(target_df, conn_matrix.columns, use_instance)
    
    # Sparse labels
    src_labels = [l if i % y_interval == 0 else '' for i, l in enumerate(src_labels)] if y_interval > 1 else src_labels
    tgt_labels = [l if i % x_interval == 0 else '' for i, l in enumerate(tgt_labels)] if x_interval > 1 else tgt_labels
    
    # Apply threshold mask
    mask = None
    vmin = None
    if threshold is not None:
        mask = conn_matrix < threshold
        vmin = threshold
        print(f"Threshold: {threshold} synapses (masking {mask.sum().sum()} of {mask.size} connections)")
    
    if not cluster:
        print(f"Source: {len(src_L)} L, {len(src_R)} R | Target: {len(tgt_L)} L, {len(tgt_R)} R")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conn_matrix, cmap=cmap, ax=ax, mask=mask, vmin=vmin,
                cbar_kws={'label': 'Synapse count'},
                xticklabels=tgt_labels, yticklabels=src_labels)
    
    # Format ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, 
                       ha='right' if x_rotation > 0 else 'center', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Add L/R boundaries (only if not clustered)
    if not cluster and len(tgt_L) > 0 and len(src_L) > 0:
        ax.axvline(x=len(tgt_L), color='red', linewidth=2, linestyle='--')
        ax.axhline(y=len(src_L), color='red', linewidth=2, linestyle='--')
    
    # Labels
    label_type = "instance" if use_instance else "bodyId"
    cluster_note = " (clustered)" if cluster else ""
    ax.set_xlabel(f'{target_name} ({label_type}){cluster_note}', fontsize=11)
    ax.set_ylabel(f'{src_name} ({label_type}){cluster_note}', fontsize=11)
    ax.set_title(f'{src_name} → {target_name} connectivity', fontsize=13)
    
    plt.tight_layout()
    return fig, ax


# %%
# =============== FETCH NEURONS ===============

print("Fetching LC10a neurons...")
LC10a_df, LC10a_roi_df, LC10a_ids = npfuncs.fetch_neuron_types(c, 'LC10a')

print("\nFetching LC14a neurons...")
LC14_df, LC14_roi_df, LC14_ids = npfuncs.fetch_neuron_types(c, 'LC14a-1')

print("\nFetching AOTU neurons...")
AOTU_df, AOTU_roi_df, AOTU_ids = npfuncs.fetch_neuron_types(c, ['AOTU019', 'AOTU025'])

# %%
# =============== PLOT LC10a → AOTU CONNECTIVITY ===============

conn_matrix, src_L, src_R, tgt_L, tgt_R = npfuncs.get_connectivity_matrix(
    LC10a_ids, AOTU_ids, LC10a_df, AOTU_df
)
#%%
fig, ax = plot_connectivity_matrix(
    conn_matrix, LC10a_df, AOTU_df, src_L, src_R, tgt_L, tgt_R,
    src_name='LC10a', target_name='AOTU', 
    x_rotation=0, y_interval=5, x_interval=1, threshold=5
)
plt.show()

# %%
# =============== PLOT LC14 → LC10a CONNECTIVITY ===============

conn_matrix_LC14, src_L, src_R, tgt_L, tgt_R = npfuncs.get_connectivity_matrix(
    LC14_ids, LC10a_ids, LC14_df, LC10a_df
)

fig, ax = plot_connectivity_matrix(
    conn_matrix_LC14, LC14_df, LC10a_df, src_L, src_R, tgt_L, tgt_R,
    src_name='LC14', target_name='LC10a', 
    x_rotation=45, y_interval=1, x_interval=5, threshold=5
)
plt.show()

# %%
# =============== PLOT LC14 → LC10a → AOTU CONNECTIVITY (TWO-HOP) ===============
# LC14 → AOTU is not a direct connection, mediated by LC10a neurons

conn_matrix_2hop, src_L, src_R, tgt_L, tgt_R = npfuncs.get_two_hop_connectivity_matrix(
    LC14_ids, LC10a_ids, AOTU_ids,
    LC14_df, LC10a_df, AOTU_df,
    weight_method='min'  # Options: 'min', 'product', 'second_hop', 'count'
)

fig, ax = plot_connectivity_matrix(
    conn_matrix_2hop, LC14_df, AOTU_df, src_L, src_R, tgt_L, tgt_R,
    src_name='LC14', target_name='AOTU (via LC10a)', 
    x_rotation=45, y_interval=1, x_interval=1, threshold=5
)
plt.show()

# %%
# =============== CLUSTERED VIEW: LC10a → AOTU ===============
# Cluster by cosine similarity to reveal functional groups

fig, ax = plot_connectivity_matrix(
    conn_matrix, LC10a_df, AOTU_df, src_L, src_R, tgt_L, tgt_R,
    src_name='LC10a', target_name='AOTU', 
    x_rotation=45, y_interval=3, x_interval=1, threshold=5, cluster=True
)
plt.show()

# %%
# =============== PLOT DENDROGRAMS ===============
# Visualize clustering hierarchy with better formatting

# Cluster the matrix
clustered_matrix, row_linkage, col_linkage = npfuncs.cluster_connectivity_matrix(conn_matrix)

# Plot dendrograms with better visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left dendrogram (LC10a neurons) - show all labels but make them sparse
# Create instance labels for LC10a neurons
lc10a_instance_map = dict(zip(LC10a_df['bodyId'], LC10a_df['instance']))
lc10a_labels = [lc10a_instance_map.get(bodyId, str(bodyId)) for bodyId in clustered_matrix.index]

# Show every 5th label to avoid overcrowding
lc10a_labels_sparse = [label if i % 5 == 0 else '' for i, label in enumerate(lc10a_labels)]

dendrogram(row_linkage, ax=ax1, orientation='left', 
           show_leaf_counts=True, leaf_rotation=0, leaf_font_size=8,
           labels=lc10a_labels_sparse)
ax1.set_title(f'LC10a neurons clustering ({len(lc10a_labels)} neurons)', fontsize=12)
ax1.set_xlabel('Distance (cosine)', fontsize=10)

# Right dendrogram (AOTU neurons) - show all since fewer
# Create instance labels for AOTU neurons
aotu_instance_map = dict(zip(AOTU_df['bodyId'], AOTU_df['instance']))
aotu_labels = [aotu_instance_map.get(bodyId, str(bodyId)) for bodyId in clustered_matrix.columns]

dendrogram(col_linkage, ax=ax2, orientation='left',
           show_leaf_counts=True, leaf_rotation=0, leaf_font_size=10,
           labels=aotu_labels)
ax2.set_title('AOTU neurons clustering', fontsize=12)
ax2.set_xlabel('Distance (cosine)', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# =============== CONNECTIVITY SUMMARY ===============
# Basic statistics about the clustered connectivity

print("=== CONNECTIVITY SUMMARY ===")
print(f"Total LC10a neurons: {len(conn_matrix)}")
print(f"Total AOTU neurons: {len(conn_matrix.columns)}")
print(f"Total connections: {(conn_matrix > 0).sum().sum()}")
avg_strength = conn_matrix.values[conn_matrix.values > 0].mean()
print(f"Average connection strength: {avg_strength:.1f} synapses")
max_strength = conn_matrix.values.max()
print(f"Max connection strength: {max_strength:.0f} synapses")

# Show top connections
print(f"\n--- Top 10 Connections ---")
top_connections = conn_matrix.stack().sort_values(ascending=False).head(10)
src_instance_map = dict(zip(LC10a_df['bodyId'], LC10a_df['instance']))
tgt_instance_map = dict(zip(AOTU_df['bodyId'], AOTU_df['instance']))
for (src, tgt), weight in top_connections.items():
    src_name = src_instance_map.get(src, str(src))
    tgt_name = tgt_instance_map.get(tgt, str(tgt))
    print(f"{src_name} ({src}) → {tgt_name} ({tgt}): {weight:.0f} synapses")

#%%
# Get connections from AOTU to DNa02
DNa02_df, DNa02_roi_df, DNa02_ids = npfuncs.fetch_neuron_types(c, 'DNa02')

conn_matrix_DNa02, src_L, src_R, tgt_L, tgt_R = npfuncs.get_connectivity_matrix(
    AOTU_ids, DNa02_ids, AOTU_df, DNa02_df
)

#%
fig, ax = plot_connectivity_matrix(
    conn_matrix_DNa02, AOTU_df, DNa02_df, src_L, src_R, tgt_L, tgt_R,
    src_name='AOTU', target_name='DNa02', 
    x_rotation=45, y_interval=1, x_interval=1, threshold=5
)
plt.show()

#%%

# Get synapse connections with fetch_synapse_connections
# First, get synapses from AOTU to DNa02
AOTU_DNa02_syn = neu.fetch_synapse_connections(AOTU_ids, DNa02_ids, nt='max')

# Get synapses from LC10a to AOTU
LC10a_AOTU_syn = neu.fetch_synapse_connections(LC10a_ids, AOTU_ids, nt='max')

# Get synapses from LC14 to LC10a
LC14_LC10a_syn = neu.fetch_synapse_connections(LC14_ids, LC10a_ids, nt='max')

#%%

# Assign excitatory or inhibitory based on neurotransmitter (nt) type
nt_lut = {'acetylcholine': 'ex', 
          'dopamine': 'dop', 
          'glutamate': 'in',
          'gaba': 'in',
          'serotonin': 'ser',
          'octopamine': 'oct',
          'histamine': 'hist'}

# Color LUT for neurotransmitter types
nt_color_lut = {'ex': 'red', 
                'dop': 'lightgray', 
                'in': 'blue', 
                'ser': 'darkgray',
                'his': 'darkgray',
                'oct': 'darkgray'}

AOTU_DNa02_syn['nt_type'] = [nt_lut[v] for v in AOTU_DNa02_syn['nt']]
LC10a_AOTU_syn['nt_type'] = [nt_lut[v] for v in LC10a_AOTU_syn['nt']]
LC14_LC10a_syn['nt_type'] = [nt_lut[v] for v in LC14_LC10a_syn['nt']]


#%%
# =============== HELPER FUNCTIONS FOR NEURAL CIRCUIT DIAGRAM ===============

def _get_default_neuron_colors():
    """Get default color scheme for neurons."""
    return {
        'LC14a-1': '#FF6B6B',    # Red
        'LC10a': '#4ECDC4',      # Teal  
        'AOTU019': '#45B7D1',    # Blue
        'AOTU025': '#96CEB4',    # Green
        'DNa02': '#F39C12'       # Orange
    }

def _calculate_neuron_positions(layers):
    """Calculate x,y positions for all neurons in layers."""
    neuron_positions = {}
    layer_y_positions = {}
    
    n_layers = len(layers)
    layer_height = 8 / n_layers
    total_width = 6
    start_x = 1
    
    for i, (layer_name, neurons) in enumerate(layers.items()):
        y_pos = 8 - (i * layer_height) - layer_height/2
        layer_y_positions[layer_name] = y_pos
        
        if len(neurons) > 1:
            spacing = total_width / (len(neurons) - 1)
            for j, neuron_id in enumerate(neurons):
                x_pos = start_x + j * spacing
                neuron_positions[neuron_id] = (x_pos, y_pos)
        else:
            neuron_positions[neurons[0]] = (4, y_pos)
    
    return neuron_positions, layer_y_positions

def _draw_neurons(ax, neuron_positions, colors):
    """Draw neuron circles with labels."""
    for neuron_id, (x, y) in neuron_positions.items():
        parts = neuron_id.split('_')
        neuron_type = '_'.join(parts[:-1])
        hemisphere = parts[-1]
        
        circle = plt.Circle((x, y), 0.3, color=colors.get(neuron_type, '#999999'), alpha=0.8, zorder=3)
        ax.add_patch(circle)
        
        ax.text(x, y, f'{neuron_type}\n{hemisphere}', ha='center', va='center', 
                fontsize=8, fontweight='bold', zorder=4)

def _get_connection_strength(src_type, src_hemisphere, tgt_type, tgt_hemisphere, matrix_name, conn_matrices, dataframes):
    """Get connection strength between specific neuron types and hemispheres."""
    if matrix_name not in conn_matrices:
        return 0
        
    conn_matrix = conn_matrices[matrix_name]
    
    # Find dataframes
    src_df = next((df for df in dataframes.values() if src_type in df['type'].values), None)
    tgt_df = next((df for df in dataframes.values() if tgt_type in df['type'].values), None)
    
    if src_df is None or tgt_df is None:
        return 0
    
    # Get neuron IDs
    src_ids = src_df[(src_df['type'] == src_type) & (src_df['somaSide'] == src_hemisphere)]['bodyId'].tolist()
    tgt_ids = tgt_df[(tgt_df['type'] == tgt_type) & (tgt_df['somaSide'] == tgt_hemisphere)]['bodyId'].tolist()
    
    # Sum connections
    total_synapses = 0
    for src_id in src_ids:
        for tgt_id in tgt_ids:
            if src_id in conn_matrix.index and tgt_id in conn_matrix.columns:
                total_synapses += conn_matrix.loc[src_id, tgt_id]
    
    return total_synapses

def _get_arrow_color(src_type, src_hemisphere, tgt_type, tgt_hemisphere, matrix_name, 
                     colors, synapse_data, nt_color_lut, dataframes):
    """Determine arrow color based on neurotransmitter type or neuron type."""
    arrow_color = colors.get(src_type, '#999999')
    
    if synapse_data and matrix_name in synapse_data and nt_color_lut:
        syn_df = synapse_data[matrix_name]
        
        # Find dataframes
        src_df = next((df for df in dataframes.values() if src_type in df['type'].values), None)
        tgt_df = next((df for df in dataframes.values() if tgt_type in df['type'].values), None)
        
        if src_df is not None and tgt_df is not None:
            src_bodyIds = src_df[(src_df['type'] == src_type) & 
                                (src_df['instance'].str.endswith(src_hemisphere))]['bodyId'].values
            tgt_bodyIds = tgt_df[(tgt_df['type'] == tgt_type) & 
                                (tgt_df['instance'].str.endswith(tgt_hemisphere))]['bodyId'].values
            
            syn_mask = syn_df['bodyId_pre'].isin(src_bodyIds) & syn_df['bodyId_post'].isin(tgt_bodyIds)
            
            if syn_mask.any():
                nt_types = syn_df[syn_mask]['nt_type'].value_counts()
                if not nt_types.empty:
                    dominant_nt = nt_types.index[0]
                    arrow_color = nt_color_lut.get(dominant_nt, arrow_color)
    
    return arrow_color

def _auto_detect_connections(layers, conn_matrices):
    """Automatically detect connections between layers."""
    layer_names = list(layers.keys())
    connections = []
    
    # Between-layer connections
    for i in range(len(layer_names) - 1):
        src_layer, tgt_layer = layer_names[i], layer_names[i + 1]
        src_types = set('_'.join(n.split('_')[:-1]) for n in layers[src_layer])
        tgt_types = set('_'.join(n.split('_')[:-1]) for n in layers[tgt_layer])
        
        matrix_name = None
        if 'LC14a-1' in src_types and 'LC10a' in tgt_types:
            matrix_name = 'LC14_to_LC10a'
        elif 'LC10a' in src_types and (src_types & {'AOTU019', 'AOTU025'}):
            matrix_name = 'LC10a_to_AOTU'
        elif (src_types & {'AOTU019', 'AOTU025'}) and 'DNa02' in tgt_types:
            matrix_name = 'AOTU_to_DNa02'
        
        if matrix_name and matrix_name in conn_matrices:
            connections.append((src_layer, tgt_layer, matrix_name))
    
    # Within-layer connections
    for layer_name in layer_names:
        layer_types = set('_'.join(n.split('_')[:-1]) for n in layers[layer_name])
        if 'LC14a-1' in layer_types and 'LC10a' in layer_types:
            connections.append((layer_name, layer_name, 'LC14_to_LC10a'))
    
    return connections

def _collect_all_connections(layers, connections, conn_matrices, dataframes, synapse_data, nt_color_lut, colors):
    """Collect all connections with their strengths and colors."""
    all_connections = []
    
    for src_layer, tgt_layer, matrix_name in connections:
        if matrix_name not in conn_matrices:
            continue
            
        for src_neuron in layers[src_layer]:
            for tgt_neuron in layers[tgt_layer]:
                if src_neuron == tgt_neuron:
                    continue
                
                # Parse neuron identifiers
                src_parts = src_neuron.split('_')
                src_type, src_hemisphere = '_'.join(src_parts[:-1]), src_parts[-1]
                tgt_parts = tgt_neuron.split('_')
                tgt_type, tgt_hemisphere = '_'.join(tgt_parts[:-1]), tgt_parts[-1]
                
                strength = _get_connection_strength(src_type, src_hemisphere, tgt_type, tgt_hemisphere, 
                                                   matrix_name, conn_matrices, dataframes)
                
                if strength > 0:
                    arrow_color = _get_arrow_color(src_type, src_hemisphere, tgt_type, tgt_hemisphere, 
                                                   matrix_name, colors, synapse_data, nt_color_lut, dataframes)
                    all_connections.append((src_neuron, tgt_neuron, strength, arrow_color))
    
    return all_connections

def _draw_arrows(ax, all_connections, neuron_positions, layers, min_width, max_width):
    """Draw arrows representing connections."""
    if not all_connections:
        return
    
    max_strength = max(conn[2] for conn in all_connections)
    same_layer_connections = {}
    circle_radius = 0.3
    
    for start, end, strength, color in all_connections:
        if start not in neuron_positions or end not in neuron_positions:
            continue
            
        x1, y1 = neuron_positions[start]
        x2, y2 = neuron_positions[end]
        
        # Find layers
        start_layer = next((name for name, neurons in layers.items() if start in neurons), None)
        end_layer = next((name for name, neurons in layers.items() if end in neurons), None)
        
        # Calculate arrow properties
        line_width = min_width + (max_width - min_width) * (strength / max_strength) if max_strength > 0 else min_width
        dx, dy = x2 - x1, y2 - y1
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 0:
            # Offset from circle edges
            offset_start = (x1 + (dx/distance)*circle_radius, y1 + (dy/distance)*circle_radius)
            offset_end = (x2 - (dx/distance)*circle_radius, y2 - (dy/distance)*circle_radius)
            
            if start_layer == end_layer:
                # Curved arrow for same-layer connections
                conn_key = tuple(sorted([start, end]))
                same_layer_connections[conn_key] = same_layer_connections.get(conn_key, 0) + 1
                offset_dir = 1 if same_layer_connections[conn_key] % 2 == 1 else -1
                
                ax.annotate('', xy=offset_end, xytext=offset_start,
                           arrowprops=dict(arrowstyle='->', lw=line_width, color=color, alpha=0.7,
                                         connectionstyle=f"arc3,rad={0.2*offset_dir}", shrinkA=0, shrinkB=0))
            else:
                # Straight arrow for between-layer connections
                ax.annotate('', xy=offset_end, xytext=offset_start,
                           arrowprops=dict(arrowstyle='->', lw=line_width, color=color, alpha=0.7,
                                         shrinkA=0, shrinkB=0))

#%
def plot_neural_circuit(layers, conn_matrices, dataframes, connections=None, synapse_data=None, nt_color_lut=None, neuron_colors=None, figsize=(14, 10), min_width=0.5, max_width=8):
    """
    Create a flexible neural circuit diagram with custom layers and connections.
    plot_neural_circuit(...)
    ├─ Setup figure and colors
    ├─ Calculate positions and draw neurons
    ├─ Auto-detect or use provided connections
    ├─ Collect all connections with colors
    ├─ Draw arrows
    └─ Add labels and title
    Parameters:
    -----------
    layers : dict
        Dictionary mapping layer names to lists of neuron identifiers
        e.g., {'Input': ['LC14a-1_L', 'LC14a-1_R'], 
               'Intermediate1': ['LC10a_L', 'LC10a_R'],
               'Intermediate2': ['AOTU019_L', 'AOTU019_R'],
               'Output': ['DNa02_L', 'DNa02_R']}
    conn_matrices : dict
        Dictionary of connectivity matrices with descriptive names
        e.g., {'LC14_to_LC10a': matrix1, 'LC10a_to_AOTU': matrix2, 'AOTU_to_DNa02': matrix3}
    dataframes : dict
        Dictionary of neuron dataframes
        e.g., {'LC14': df1, 'LC10a': df2, 'AOTU': df3, 'DNa02': df4}
    connections : list, optional
        List of (src_layer, tgt_layer, matrix_name) tuples to specify which layers connect
        e.g., [('Input', 'Intermediate1', 'LC14_to_LC10a'), 
               ('Intermediate1', 'Intermediate2', 'LC10a_to_AOTU'),
               ('Intermediate2', 'Output', 'AOTU_to_DNa02')]
        If None, will connect all adjacent layers automatically
    synapse_data : dict, optional
        Dictionary of synapse dataframes with neurotransmitter information
        e.g., {'LC14_to_LC10a': syn_df1, 'LC10a_to_AOTU': syn_df2}
    nt_color_lut : dict, optional
        Dictionary mapping neurotransmitter types to colors
        e.g., {'ex': 'red', 'in': 'blue', 'dop': 'gray'}
    neuron_colors : dict, optional
        Dictionary mapping neuron types to colors for the node circles
        e.g., {'LC14a-1': '#FF6B6B', 'LC10a': '#4ECDC4'}
        If None, uses default color scheme
    figsize : tuple
        Figure size
    min_width, max_width : float
        Min/max arrow width for scaling
    """
    # Setup
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Get colors
    colors = neuron_colors if neuron_colors is not None else _get_default_neuron_colors()
    
    # Calculate positions and draw neurons
    neuron_positions, layer_y_positions = _calculate_neuron_positions(layers)
    _draw_neurons(ax, neuron_positions, colors)
    
    # Auto-detect or use provided connections
    if connections is None:
        connections = _auto_detect_connections(layers, conn_matrices)
    
    # Collect all connections with strengths and colors
    all_connections = _collect_all_connections(layers, connections, conn_matrices, dataframes, 
                                              synapse_data, nt_color_lut, colors)
    
    # Draw all arrows
    _draw_arrows(ax, all_connections, neuron_positions, layers, min_width, max_width)
    
    # Add hemisphere labels (positioned at edges of the layout)
    ax.text(1, 9.5, 'LEFT', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkred')
    ax.text(7, 9.5, 'RIGHT', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkred')
    
    
    # Add title
    ax.text(5, 9.8, 'Neural Circuit Connectivity', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

# %%
# =============== NEURAL CIRCUIT DIAGRAM ===============
# Define custom layers

# Example 1: Separate AOTU types into different layers
layers_separate = {
    'Input': ['LC14a-1_L', 'LC14a-1_R'],
    'Intermediate': ['LC10a_L', 'LC10a_R'],
    'Output AOTU019': ['AOTU019_L', 'AOTU019_R'],
    'Output AOTU025': ['AOTU025_L', 'AOTU025_R']
}

# Example 2: Group AOTU types on same layer
layers_grouped = {
    'Input': ['LC14a-1_L', 'LC14a-1_R'],
    'Intermediate': ['LC10a_L', 'LC10a_R'],
    'Output': ['AOTU019_L', 'AOTU019_R', 'AOTU025_L', 'AOTU025_R']
}

# Connectivity matrices and dataframes
conn_matrices = {
    'LC14_to_LC10a': conn_matrix_LC14,
    'LC10a_to_AOTU': conn_matrix,
    'AOTU_to_DNa02': conn_matrix_DNa02
}

dataframes = {
    'LC14': LC14_df,
    'LC10a': LC10a_df, 
    'AOTU': AOTU_df,
    'DNa02': DNa02_df
}

# Synapse data with neurotransmitter information
synapse_data = {
    'LC14_to_LC10a': LC14_LC10a_syn,
    'LC10a_to_AOTU': LC10a_AOTU_syn,
    'AOTU_to_DNa02': AOTU_DNa02_syn
}

# Neuron colors for the diagram nodes
neuron_colors = {
    'LC14a-1': '#FF6B6B',    # Red
    'LC10a': '#4ECDC4',      # Teal  
    'AOTU019': '#45B7D1',    # Blue
    'AOTU025': '#96CEB4',    # Green
    'DNa02': '#F39C12'       # Orange
}

# Debug: Check actual neuron types in dataframes
print("=== DEBUGGING NEURON TYPES ===")
for df_name, df in dataframes.items():
    print(f"{df_name} dataframe types: {df['type'].unique()}")

# Example 1: 4-layer circuit with LC14a-1 and LC10a on same layer
layers_4layer = {
    'Input': ['LC14a-1_L', 'LC10a_L', 'LC10a_R', 'LC14a-1_R' ],  # Combined input layer
    'Intermediate': ['AOTU025_L', 'AOTU019_L', 'AOTU019_R', 'AOTU025_R'],  # Custom order
    'Output': ['DNa02_L', 'DNa02_R']
}

# Define explicit connections (including within-layer connections)
connections_3layer = [
    ('Input', 'Input', 'LC14_to_LC10a'),        # LC14a-1 to LC10a within same layer
    ('Input', 'Intermediate', 'LC10a_to_AOTU'), # LC10a connects to AOTU
    ('Intermediate', 'Output', 'AOTU_to_DNa02') # AOTU connects to DNa02
]

# Plot 3-layer circuit with neurotransmitter colors
print("\n=== 3-LAYER CIRCUIT (LC14a-1 + LC10a combined) WITH NEUROTRANSMITTER COLORS ===")
fig, ax = plot_neural_circuit(layers_4layer, conn_matrices, dataframes, connections_3layer, 
                             synapse_data=synapse_data, nt_color_lut=nt_color_lut, 
                             neuron_colors=neuron_colors)
plt.show()


# %%

#LC10a_syn = neu.fetch_synapses(LC10a_ids, nt='max')
#%%
#AOTU_syn = neu.fetch_synapses(AOTU_ids, nt='max')

LC10a_AOTU_syn['side'] = LC10a_AOTU_syn['roi_pre'].map(lambda x: 'L' if 'L' in x else 'R')
print(LC10a_AOTU_syn[['roi_pre', 'roi_post', 'side']])

#%%
#LC10a_syn['nt_type'] = [nt_lut[v] for v in LC10a_syn['nt']]
#LC10a_syn.head()
# %%
# Plot pre and post synapses
xvar = 'x'
yvar = 'y'

curr_syn = LC10a_AOTU_syn[LC10a_AOTU_syn['side']=='R']

fig, ax = plt.subplots(figsize=(10, 5))
#sns.scatterplot(data=LC10a_syn[LC10a_syn['type']=='pre'], ax=ax, 
sns.scatterplot(data=curr_syn, ax=ax, 
            x=f'{xvar}_pre', y=f'{yvar}_pre', alpha=0.5,
            hue='z_pre', palette='viridis_r') #hue='nt_type', palette=nt_color_lut)
#ax.set_aspect('equal')

#%
AOTU019_ids = AOTU_df[AOTU_df['type']=='AOTU019']['bodyId'].unique()
AOTU025_ids = AOTU_df[AOTU_df['type']=='AOTU025']['bodyId'].unique()

curr_syn1 = curr_syn[curr_syn['bodyId_post'].isin(AOTU019_ids)]
curr_syn2 = curr_syn[curr_syn['bodyId_post'].isin(AOTU025_ids)]
#sns.scatterplot(data=curr_syn1, ax=ax,
#            x=f'{xvar}_post', y=f'{yvar}_post', 
#            color='red', s=10, alpha=0.5)
sns.scatterplot(data=curr_syn2, ax=ax,
            x=f'{xvar}_post', y=f'{yvar}_post',
            color='blue', s=10, alpha=0.5)
            

#%%
AOTU019_ids = AOTU_df[AOTU_df['type']=='AOTU019']['bodyId'].unique()
print(len(AOTU019_ids))
#%
aotu_plot = AOTU_syn[AOTU_syn['bodyId'].isin(AOTU019_ids)]

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=aotu_plot[aotu_plot['type']=='pre'], ax=ax, 
            x='z', y='y', hue='z', palette='viridis_r') #hue='nt_type', palette=nt_color_lut)
ax.set_aspect('equal')
# %%
