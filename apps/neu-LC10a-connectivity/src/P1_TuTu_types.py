'''
 # @ Author: Juliana Rhee
 # @ Filename: P1_TuTu_types.py
 # @ Create Time: 2025-10-13 10:53:42
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-10-13 10:53:48
 # @ Description: Plot the connectivity of P1 TuTu types to LC10a neurons
 '''

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.compat import F
import seaborn as sns

import neuprint as neu
from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import utils as util

#%%

def plot_connection_matrix(conn_matrix, 
                           show_all_row_labels=False,
                           show_all_col_labels=False,
                           normalize_colors=True,
                           vmin=10, vmax=None,
                           colorbar_label='weight',
                           figsize=None,
                           ax=None):

    if ax is None:
        # Auto-adjust figure size if showing all labels
        if figsize is None:
            if show_all_row_labels or show_all_col_labels:
                # Calculate size based on number of labels
                n_rows = len(conn_matrix.index)
                n_cols = len(conn_matrix.columns)
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
                cbar_kws={'shrink': 0.5, 'anchor': (0, 0.0), 'label': colorbar_label},
                yticklabels=yticklabels, xticklabels=xticklabels)
    
    # Modify colorbar tick labels if vmax is provided and not 1.0
    if vmax is not None and vmax != 1.0:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            # Get current tick labels using the correct method
            tick_labels = cbar.ax.get_yticklabels()
            if tick_labels:
                # Get max tick_label value
                print("Replacing tick labels")
                max_tick_label = max([float(label.get_text()) for label in tick_labels])
                print(max_tick_label)
                tick_labels[-1] = f'>{vmax:.2f}'
                cbar.ax.set_yticklabels(tick_labels)
        else:
            print("No colorbar found")
    # Adjust tick label font size when showing all labels
    if show_all_row_labels:
        ax.tick_params(axis='y', labelsize=6)
    if show_all_col_labels:
        ax.tick_params(axis='x', labelsize=6, rotation=90)

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
                                   figsize=None,
                                   normalize_colors=True,
                                   vmin=10,
                                   colorbar_label='weight',
                                   ax=None):
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
        If True, show all row tick labels
    show_all_col_labels : bool, default False
        If True, show all column tick labels
    figsize : tuple, optional
        Figure size (width, height). If None, uses (10, 10) or auto-adjusts for all labels
    normalize_colors : bool, default True
        If True, normalize the colorbar to use the full range of the data
    colorbar_label : str, default 'weight'
        Label for the colorbar
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        # Auto-adjust figure size if showing all labels
        if figsize is None:
            if show_all_row_labels or show_all_col_labels:
                # Calculate size based on number of labels
                n_rows = len(conn_matrix.index)
                n_cols = len(conn_matrix.columns)
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
        # Use the actual data range for better color contrast
        vmin = conn_matrix.min().min() if not conn_matrix.empty else 0
        vmax = conn_matrix.max().max() if not conn_matrix.empty else 1
        # Skip very low values to improve contrast
        if vmin < 0.1 * vmax:
            vmin = 0.1 * vmax
    else:
        vmin = vmin
        vmax = None
    
    sns.heatmap(conn_matrix, ax=ax, vmin=vmin, vmax=vmax, cmap='magma',
                cbar_kws={'shrink': 0.5, 'anchor': (0, 0.0), 'label': colorbar_label},
                yticklabels=yticklabels, xticklabels=xticklabels)
    
    # Adjust tick label font size when showing all labels
    if show_all_row_labels:
        ax.tick_params(axis='y', labelsize=6)
    if show_all_col_labels:
        ax.tick_params(axis='x', labelsize=6, rotation=90)

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
            label_offset = xlim[0] - 0.03 * plot_width  # 3% of plot width to the left
            
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

    return fig


def highlight_row_or_column(ax, conn_matrix, row_label=None, column_label=None, 
                           color='white', linewidth=2, highlight_label=True, highlight_label_color='red'):
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


def norm_conn_matrix_by_target_inputs(conn_matrix, conn_df, target='instance_post'):
    neuron_, conn_ = neu.fetch_adjacencies(targets=conn_df[target].unique())
    all_inputs_to_targets = conn_.groupby('bodyId_post', as_index=False)['weight'].sum()
    
    target_type = target.split('_')[0]
    all_inputs_to_targets[target] = all_inputs_to_targets['bodyId_post'].apply(lambda x: neuron_.loc[neuron_['bodyId']==x, target_type].values[0])
    for col in conn_matrix.columns:
        total_weight = all_inputs_to_targets[all_inputs_to_targets[target]==col]['weight'].values[0]
        conn_matrix[col] = conn_matrix[col].div(total_weight)
    return conn_matrix

#%%
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
c = Client('neuprint.janelia.org', dataset='male-cns:v0.9', token=token)
c.fetch_version()

#%%
# Get all LC10a neurons
# ======================================================
LC10a_neurons, LC10a_roi_counts = neu.fetch_neurons(NC(type='LC10a', client=c))
# Assign sign
LC10a_sides = neu.assign_sides_in_groups(LC10a_neurons, LC10a_roi_counts)
LC10a_sides.loc[10573]

#LC10a_neurons
# %%
# Get all inputs
# -------------------------------------------------------
min_total_weight = 10
# bodyId_pre are INPUTS to targets, bodyId_post are the target IDs
LC10a_inputs_neuron_df, LC10a_inputs_conn_df = neu.fetch_adjacencies(targets=NC(type=['LC10a']),
                                                min_total_weight=min_total_weight)
LC10a_inputs_conn_df = neu.merge_neuron_properties(LC10a_inputs_neuron_df, LC10a_inputs_conn_df, ['type', 'instance'])
LC10a_inputs_conn_df['side'] = LC10a_inputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
# Remove side from roi
LC10a_inputs_conn_df['roi_noside'] = LC10a_inputs_conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)

#%
# LC10a: Group conn_df by type_pre, and sort by sum of weight
sorted_LC10a_inputs = LC10a_inputs_conn_df.groupby(['roi_noside', 
                                                    'type_pre'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('LC10a inputs:')
print(sorted_LC10a_inputs.iloc[0:20])

#%%
# LC10a: Get all outputs: 
# -------------------------------------------------------
# bodyId_pre are the LC10a source IDs, bodyId_post are the target IDs
LC10a_outputs_neuron_df, LC10a_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type=['LC10a']), 
                                                        targets=None,
                                                        min_total_weight=min_total_weight)
LC10a_outputs_conn_df = neu.merge_neuron_properties(LC10a_outputs_neuron_df, LC10a_outputs_conn_df, ['type', 'instance'])
LC10a_outputs_conn_df['side'] = LC10a_outputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
# Remove side from roi
LC10a_outputs_conn_df['roi_noside'] = LC10a_outputs_conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
# Sort 
sorted_LC10a_outputs = LC10a_outputs_conn_df.groupby(['roi_noside', 
                                                      'type_post'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('LC10a outputs:')
print(sorted_LC10a_outputs.iloc[0:20])
#%%

# Show Connection Matrix for LC10a
separate_by_side = True
use_log_weights = True

pre_variable = 'type_pre'
post_variable = 'bodyId_post'

sorted_by_grouper = True 
pre_grouper = 'roi_noside'
post_grouper = 'type_post'
LC10a_in = LC10a_inputs_conn_df[LC10a_inputs_conn_df['side']=='R'].copy()

LC10a_in_conn_matrix = connection_table_to_matrix(LC10a_in,
                        group_cols=[pre_variable, post_variable],
                        sort_by= [ pre_grouper, post_grouper]) #, 'bodyId', sort_by='instance')
                    
if not sorted_by_grouper:
    # Sort by weight
    assert pre_variable == 'type_pre'
    sorted_in= LC10a_in.groupby([pre_variable])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
    LC10a_in_conn_matrix = LC10a_in_conn_matrix.loc[sorted_in[pre_variable].values]
    #LC10a_in_conn_matrix = LC10a_in_conn_matrix[sorted_in['type_post'].values] 

# ROI colors
n_pre_groups = len(LC10a_in[pre_grouper].unique())
n_post_groups = len(LC10a_in[post_grouper].unique())
pre_grouper_dict = {roi: sns.color_palette("colorblind", n_pre_groups)[i] 
                    for i, roi in enumerate(LC10a_in[pre_grouper].unique())}
post_grouper_dict = {roi: sns.color_palette("colorblind", n_post_groups)[i] 
                    for i, roi in enumerate(LC10a_in[post_grouper].unique())}

# Plot LC10a inputs - annotate rows (pre/inputs)
if use_log_weights:
    LC10a_conn = util.log_weights(LC10a_in_conn_matrix)
    colorbar_label = 'log(weight)'
else:
    LC10a_conn = LC10a_in_conn_matrix
    colorbar_label = 'weight'
fig = plot_grouped_connection_matrix(LC10a_conn, LC10a_in, 
                                     pre_grouper_dict=pre_grouper_dict,
                                     post_grouper_dict=post_grouper_dict,
                                     group_per_row=None,
                                     group_per_col=None,
                                     pre_grouper = pre_grouper,
                                     post_grouper = post_grouper,
                                     sorted_by_grouper=sorted_by_grouper,
                                     pre_variable=pre_variable,
                                     post_variable=post_variable,
                                     annotate_rows=True,
                                     annotate_cols=True,
                                     show_all_row_labels=True,
                                     show_all_col_labels=False,
                                     colorbar_label='log(weight)')
fig.axes[0].set_title('LC10a inputs')

highlight_row_or_column(fig.axes[0], LC10a_conn, row_label='TuTuA_2',
                        color='k', linewidth=2)

#%%
# LC10 inputs: aggregate side and ROI
separate_by_side = False
#use_log_weights = True
weight_type = 'percent' # can be: 'log', 'percent', 'weight'

pre_variable = 'type_pre'
post_variable = 'bodyId_post'
sorted_by_grouper = False
pre_grouper = 'type_pre'
post_grouper = 'type_post'
LC10a_in = LC10a_inputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)

LC10a_in_conn_matrix = connection_table_to_matrix(LC10a_in,
                        group_cols=[pre_variable, post_variable],
                        sort_by= [ pre_grouper, post_grouper]) #, 'bodyId', sort_by='instance')
                    
if not sorted_by_grouper:
    # Sort by weight
    assert pre_variable == 'type_pre'
    sorted_in= LC10a_in.groupby([pre_variable])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
    LC10a_in_conn_matrix = LC10a_in_conn_matrix.loc[sorted_in[pre_variable].values]
    #LC10a_in_conn_matrix = LC10a_in_conn_matrix[sorted_in['type_post'].values] 

# Plot LC10a inputs - annotate rows (pre/inputs)
vmin=None; vmax=None;
if weight_type == 'log':
    LC10a_conn = util.log_weights(LC10a_in_conn_matrix)
    colorbar_label = 'log(weight)'
elif weight_type == 'percent':
    LC10a_conn = norm_conn_matrix_by_target_inputs(LC10a_in_conn_matrix, LC10a_in, 
                                                  target='bodyId_post')
    LC10a_conn[LC10a_conn==0] = np.nan
    colorbar_label = 'percent of total inputs'
    vmin = 0
    vmax = 0.1
else:
    LC10a_conn = LC10a_in_conn_matrix
    colorbar_label = 'weight'

fig, ax = plt.subplots(figsize=(12, 10))
fig = plot_connection_matrix(LC10a_conn, ax=ax,
                             vmin=vmin, vmax=vmax,
                             show_all_row_labels=True,
                             show_all_col_labels=False,
                             colorbar_label=colorbar_label,
                             normalize_colors=True)
ax.set_title('LC10a inputs')


#%%
# LC10a OUTPUTS: 
pre_variable = 'bodyId_pre'
post_variable = 'type_post'

sorted_by_grouper = False
pre_grouper = 'roi_noside'
post_grouper = 'roi_noside'
               
# Get conn matrix                    
LC10a_out = LC10a_outputs_conn_df[LC10a_outputs_conn_df['side']=='R'].copy()
LC10a_out_conn_matrix = connection_table_to_matrix(LC10a_out,
                        group_cols=['bodyId_pre', 'type_post'],
                        sort_by= ['weight', post_grouper])#'weight']) #, 'bodyId', sort_by='instance')    
if not sorted_by_grouper:
    # Sort by weight
    sorted_out = LC10a_out.groupby(['type_post'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
    LC10a_out_conn_matrix = LC10a_out_conn_matrix[sorted_out['type_post'].values]

# Colormap
post_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(LC10a_out['roi_noside'].unique())}

# Plot LC10a outputs - annotate columns (post/outputs)
log_LC10a_out = np.log(LC10a_out_conn_matrix)
# Replace inf with 0
log_LC10a_out = log_LC10a_out.replace(np.inf, np.nan)
log_LC10a_out = log_LC10a_out.replace(-np.inf, np.nan)

fig = plot_grouped_connection_matrix(log_LC10a_out, 
                                     LC10a_out, 
                                     post_grouper_dict=post_grouper_dict,
                                     sorted_by_grouper=sorted_by_grouper,
                                     pre_grouper = pre_grouper,
                                     post_grouper = post_grouper,
                                     pre_variable=pre_variable,
                                     post_variable=post_variable,
                                     annotate_rows=True,
                                     annotate_cols=True, 
                                     show_all_col_labels=True,
                                     colorbar_label='log(weight)')
fig.axes[0].set_title('LC10a outputs')
#%
# Add a thin box around a specified column or row based on the label
highlight_row_or_column(fig.axes[0], log_LC10a_out, 
                        column_label=['AOTU019', 'AOTU025', 'P1_1b'],
                        color='k', linewidth=2)


# %%
# Get all TuTuA_2 neurons
# ======================================================
TuTuA2_neurons, TuTuA2_roi_counts = neu.fetch_neurons(NC(type='TuTuA_2', 
                                                         client=c))
#%
# Get all inputs
TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df = neu.fetch_adjacencies(targets=NC(type=['TuTuA_2']))
TuTuA2_inputs_conn_df = neu.merge_neuron_properties(TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df, ['type', 'instance'])
TuTuA2_inputs_conn_df['side'] = TuTuA2_inputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
# 
# TuTuA_2: Group conn_df by type_pre, and sort by sum of weight
sorted_TuTuA2_inputs = TuTuA2_inputs_conn_df.groupby(['roi', 'type_pre', 'side'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('TuTuA_2 inputs:')
print(sorted_TuTuA2_inputs.iloc[0:20])

#%
# Get all outputs
TuTuA2_outputs_neuron_df, TuTuA2_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type=['TuTuA_2']), targets=None)
TuTuA2_outputs_conn_df = neu.merge_neuron_properties(TuTuA2_outputs_neuron_df, TuTuA2_outputs_conn_df, ['type', 'instance'])
TuTuA2_outputs_conn_df['side'] = TuTuA2_outputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
#
# TuTuA_2: Group conn_df by type_post, and sort by sum of weight
sorted_TuTuA2_outputs = TuTuA2_outputs_conn_df.groupby(['roi', 'type_post', 'side'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('TuTuA_2 outputs:')
print(sorted_TuTuA2_outputs.iloc[0:20])

#%%
# TuTuA_2 inputs: Aggregate all weights (aggregate across ROIs) to get total connection weights
# ------------------------------------------------------------
TuTuA2_in = TuTuA2_inputs_conn_df[(TuTuA2_inputs_conn_df['weight']>=10  )]\
                  .groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'instance_post'],
                  as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)
TuTuA2_in_conn_mat = connection_table_to_matrix(TuTuA2_in,
                        group_cols=['type_pre', 'instance_post'],
                        sort_by= ['weight', 'weight'])

# Normalize by dividing each column in matrix by total weight of each post-target
TuTuA2_in_conn_mat = norm_conn_matrix_by_target_inputs(TuTuA2_in_conn_mat, TuTuA2_in, 
                                        target='instance_post')
#%
# Plot TuTuA_2 inputs: aggregate side and ROI
use_log_weights = False 
if use_log_weights:
    TuTuA2_in_conn_log = util.log_weights(TuTuA2_in_conn_mat)
    vmax = TuTuA2_in_conn_log.max().max()
    vmin = TuTuA2_in_conn_log.min().min()
    colorbar_label = 'log(weight)'
else:
    TuTuA2_in_conn_log = TuTuA2_in_conn_mat
    vmin=0.0
    vmax=0.1
    colorbar_label = 'weight'
fig, ax = plt.subplots(figsize=(6, 6))
fig = plot_connection_matrix(TuTuA2_in_conn_log, ax=ax, 
                             vmin=vmin, vmax=vmax,
                             show_all_row_labels=True,
                             show_all_col_labels=True,
                             colorbar_label=colorbar_label,
                             normalize_colors=True)
ax.set_title('TuTuA_2 inputs')
ax.set_xlabel('Post-synaptic bodyId')
ax.set_ylabel('Pre-synaptic bodyId')
plt.show()
#
#%% 
# Plot TuTuA_2 inputs: Separate by ROI/side
# ------------------------------------------------------------
sort_weights = False #True
plot_by_side = False
use_log_weights = True

if plot_by_side:
    pre_variable = 'instance_pre'
    pre_grouper = 'side' 
    sorted_by_grouper = True 
    sort_weights = False
else:
    pre_variable = 'type_pre'    
    pre_grouper = 'roi_noside'
    sorted_by_grouper = sort_weights is False

post_variable = 'instance_post'
manual_groups = sorted_by_grouper #False
post_grouper = 'roi_noside' #'type_post'
# --------
TuTuA2_inputs_conn_df['roi_noside'] = TuTuA2_inputs_conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
TuTuA2_in = TuTuA2_inputs_conn_df[#(TuTuA2_inputs_conn_df['side']=='R')
                            (TuTuA2_inputs_conn_df['weight']>=10)].copy()
#%
TuTuA2_in_conn_matrix = connection_table_to_matrix(TuTuA2_in,
                        group_cols=[pre_variable, post_variable],
                        sort_by= [ pre_grouper, post_grouper]) #'weight']) 

if manual_groups: #sorted_by_grouper:
    in_vals = TuTuA2_in_conn_matrix.index.tolist()
    sort_by_roi = TuTuA2_in[[pre_variable, pre_grouper]]\
                            .drop_duplicates()\
                            .sort_values(by=pre_grouper)
    TuTuA2_in_conn_matrix = TuTuA2_in_conn_matrix.loc[sort_by_roi[pre_variable].values]
    group_per_row = sort_by_roi[pre_grouper].values
else:
    group_per_row = None

    # Sort by weight
    sorted_TuTuA2_inputs = TuTuA2_in.groupby([ pre_variable, pre_grouper])['weight'].sum().reset_index().sort_values(by=['weight'], ascending=False)
    sorted_TuTuA2_inputs = TuTuA2_in.groupby(['type_pre'])['weight'].sum().reset_index().sort_values(by=['weight'], ascending=False)
    TuTuA2_in_conn_matrix = TuTuA2_in_conn_matrix.loc[sorted_TuTuA2_inputs['type_pre'].values]

# ROI colors
pre_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(TuTuA2_in[pre_grouper].unique())}
post_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(TuTuA2_in[post_grouper].unique())}

#% Plot TuTuA_2 inputs
if use_log_weights:
    plot_TuTuA2_in = util.log_weights(TuTuA2_in_conn_matrix)
    colorbar_label = 'log(weight)'
else:
    plot_TuTuA2_in = TuTuA2_in_conn_matrix
    colorbar_label = 'weight'

fig = plot_grouped_connection_matrix(plot_TuTuA2_in, TuTuA2_in, 
                                     pre_grouper_dict=pre_grouper_dict,
                                     post_grouper_dict=post_grouper_dict,
                                     pre_grouper = pre_grouper,
                                     post_grouper = post_grouper,
                                     sorted_by_grouper=sorted_by_grouper,
                                     group_per_row = group_per_row,
                                     pre_variable=pre_variable,
                                     post_variable=post_variable,
                                     annotate_rows=True,
                                     annotate_cols=True,
                                     show_all_row_labels=True,
                                     show_all_col_labels=True,
                                     colorbar_label=colorbar_label)
fig.axes[0].set_title('TuTuA_2 inputs')

# Highlight
highlight_row_or_column(fig.axes[0], plot_TuTuA2_in, row_label='SMP054',
                        color='k', linewidth=2)

#%%
# Get all P1 types
# ======================================================
P1_types = ['P1_1b']
P1_1a_inputs_neuron_df, P1_1a_inputs_conn_df = neu.fetch_adjacencies(targets=NC(type=P1_types, 
                                            client=c))
P1_1a_inputs_conn_df = neu.merge_neuron_properties(P1_1a_inputs_neuron_df, P1_1a_inputs_conn_df, ['type', 'instance'])
P1_1a_inputs_conn_df['side'] = P1_1a_inputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
#%
# P1_1a: Group conn_df by type_pre, and sort by sum of weight
sorted_P1_1a_inputs = P1_1a_inputs_conn_df.groupby(['type_post', 
                                                    'type_pre', 
                                                    'side'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('P1_1a inputs:')
print(sorted_P1_1a_inputs.iloc[0:20])
#%
# Get all P1 outputs
P1_1a_outputs_neuron_df, P1_1a_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type=P1_types), targets=None)
P1_1a_outputs_conn_df = neu.merge_neuron_properties(P1_1a_outputs_neuron_df, P1_1a_outputs_conn_df, ['type', 'instance'])
P1_1a_outputs_conn_df['side'] = P1_1a_outputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
#%
# P1_1a: Group conn_df by type_post, and sort by sum of weight
sorted_P1_1a_outputs = P1_1a_outputs_conn_df.groupby(['type_post', 
                                                    'type_pre', 
                                                    'side'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('P1_1a outputs:')
print(sorted_P1_1a_outputs.iloc[0:20])
#%%
# What are the P1 inputs to SMP/SIP?
P1_types = ['P1_1b', 'P1_1a', 'P1_4a', 'P1_4b', 'P1_18a', 'P1_18b']
SMP_types = ['SMP054', 'SMP391', 'SMP394']
neu.fetch_adjacencies(sources=NC(type=P1_types, client=c),
                      targets=NC(type=SMP_types, client=c))

# %%

#%%
# Plot connection matrix
pre_type = 'P1'
post_type = 'LC'

# Get ALL connections between all P1 types and all LC types
neuron_df, conn_df = neu.fetch_adjacencies(sources=NC(type=f'{pre_type}.*', client=c),
                                           targets=NC(type=f'{post_type}.*', client=c))
conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
conn_df['side'] = conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)

print(conn_df.shape)

curr_conns = conn_df.copy()
sort_by = 'type'
weight_type = 'percent' # = False
curr_conn_matrix = connection_table_to_matrix(curr_conns,
                        group_cols=['type_pre', 'type_post'],
                        sort_by= ['type_pre', 'type_post'])
if sort_by == 'type':
    # Apply natsort to each string in the array
    pre_order = sorted(curr_conns['type_pre'].unique(), key=util.natsort)
    post_order = sorted(curr_conns['type_post'].unique(), key=util.natsort)
    curr_conn_matrix = curr_conn_matrix.reindex(index=pre_order, columns=post_order)
 
# Use log weights
if weight_type=='log':
    curr_conn_matrix = util.log_weights(curr_conn_matrix)
    # Adjust colorbar range to match data
    vmax = curr_conn_matrix.max().max()
    vmin = curr_conn_matrix.min().min()
    colorbar_label = 'log(weight)'
elif weight_type == 'percent':
    # Normalize by inputs
    curr_conn_matrix = norm_conn_matrix_by_target_inputs(curr_conn_matrix, curr_conns, 
                                                      target='type_post')
    colorbar_label = 'percent of total'
    vmin = 0.0
    vmax = 0.1
else:
    colorbar_label = 'weight'
    
# Include colorbar title
fig, ax = plt.subplots(figsize=(6, 6))
plot_connection_matrix(curr_conn_matrix, ax=ax,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True)
ax.set_title('{} -> {} connections'.format(pre_type, post_type))
ax.set_xlabel('Post-synaptic {} type'.format(post_type))
ax.set_ylabel('Pre-synaptic {} type'.format(pre_type))
plt.show()
#%%