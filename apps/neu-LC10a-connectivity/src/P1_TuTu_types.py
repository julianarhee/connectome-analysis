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
#from pandas.compat import F
import seaborn as sns

import neuprint as neu
from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import utils as util
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform

#%%

def plot_connection_matrix(conn_matrix, 
                           show_all_row_labels=False,
                           show_all_col_labels=False,
                           normalize_colors=True,
                           vmin=10, vmax=None,
                           colorbar_label='weight',
                           figsize=None,
                           show_grid=False,
                           grid_lw=0.5,
                           grid_color='white',
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
                cbar_kws={'shrink': 0.1, 'anchor': (0, 0.0), 'label': colorbar_label},
                yticklabels=yticklabels, xticklabels=xticklabels,
                linewidths=grid_lw if show_grid else 0,
                linecolor=grid_color if show_grid else None)
    
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
    row_distance = 1 - row_similarity  # Convert similarity to distance
    np.fill_diagonal(row_distance, 0)  # Ensure diagonal is exactly zero
    row_linkage = linkage(squareform(row_distance), method=method)
    
    # Calculate cosine similarity for columns (target patterns)
    col_similarity = cosine_similarity(matrix_filled.values.T)
    col_distance = 1 - col_similarity  # Convert similarity to distance
    np.fill_diagonal(col_distance, 0)  # Ensure diagonal is exactly zero
    col_linkage = linkage(squareform(col_distance), method=method)
    
    return row_linkage, col_linkage


def cluster_matrix_cosine_similarity(conn_matrix, method='ward', threshold=None, threshold_percentile=None):
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


def cluster_matrix_cosine_similarity_thresholded(conn_matrix, method='ward', threshold=None, threshold_percentile=90):
    """
    Convenience function for thresholded clustering. Calls cluster_matrix_cosine_similarity with thresholding.
    
    Parameters:
    -----------
    conn_matrix : pd.DataFrame
        Connection matrix to cluster
    method : str, default 'ward'
        Linkage method for hierarchical clustering
    threshold : float, optional
        Absolute threshold for connections (e.g., 0.05). Higher values keeps most connections (conservative, 90). Lower (e.g., 50) keeps only strongest connections.
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

# sort index
def sort_index(conn_matrix, conn_df=None, axis='rows', sort_by=None):
    '''
    Sort rows or columns of a connection matrix by weight or ROI.
    Args:
        conn_matrix: DataFrame, the connection matrix to sort
        conn_df: DataFrame, the connection dataframe
        axis: str, 'rows' or 'cols', the axis to sort
        sort_by: str, 'weight' or 'roi', the column to sort by (or None to not sort)
    Returns:
        sorted_ix: list, the sorted indices
    '''
    if axis == 'rows':
        summed_axis = 1 # Sum across columns, get total weight for each row
    else:
        summed_axis = 0 # Sum across rows, get total weight for each column 
        
    if sort_by == 'weight':
        sorted_ix = conn_matrix.sum(axis=summed_axis).sort_values(ascending=False).index.tolist()
    elif axis == 'roi':
        assert conn_df is not None, "conn_df is required for sorting by ROI"
        sorted_ix = conn_df.sort_values(by=sort_by).index.tolist()
    else:
        raise ValueError(f"Invalid sorting axis: {sort_by}")

    return sorted_ix

def sort_matrix_labels(conn_matrix, conn_df=None, 
                       sort_rows=None, sort_cols=None):
    '''
    Sort the rows and columns of a connection matrix.
    Args:
        conn_matrix: DataFrame, the connection matrix to sort
        conn_df: DataFrame, the connection dataframe
        sort_rows: str, the column to sort rows by, or None to not sort rows
        sort_cols: str, the column to sort columns by, or None to not sort columns
    Returns:
        conn_matrix: DataFrame, the sorted connection matrix
    '''
    sorted_rows = None
    sorted_cols = None
    if sort_rows is not None:
        if isinstance(sort_rows, (list, np.ndarray)):
            sorted_rows = sort_rows
        else: # sort by string (column in conn_df)
            sorted_rows = sort_index(conn_matrix, conn_df=conn_df, axis='rows', sort_by=sort_rows)
    if sort_cols is not None:
        if isinstance(sort_cols, (list, np.ndarray)):
            sorted_cols = sort_cols
        else: # sort by string (column in conn_df)  
            sorted_cols = sort_index(conn_matrix, conn_df=conn_df, axis='cols', sort_by=sort_cols)
        
    conn_matrix = conn_matrix.reindex(index=sorted_rows, columns=sorted_cols)
    
    return conn_matrix


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
                                       sort_rows=sort_rows, 
                                       sort_cols=intermediate_neurons)
    
    conn_matrix2 = sort_matrix_labels(conn_matrix2, conn_df=conn_df2, 
                                       sort_rows=intermediate_neurons, 
                                       sort_cols=sort_cols)
    conn_combined = conn_matrix1.dot(conn_matrix2)
    if return_all:
        return conn_matrix1, conn_matrix2, conn_combined
    else:
        return conn_combined

def normalize_weights_by_total(conn_df, group_col='instance_post'):
    '''
    Normalize the weights of a connection dataframe by the total weights of a given group.
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
LC10a_outputs_neuron_df, LC10a_outputs_conn_df = neu.fetch_adjacencies(
                                                        sources=NC(type=['LC10a']), 
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
TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                                       targets=NC(type=['TuTuA_2']))
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
weight_type = 'percent_of_total' # can be: 'weight', 'percent', 'log'

TuTuA2_in = TuTuA2_inputs_conn_df[(TuTuA2_inputs_conn_df['weight']>=10  )]\
                  .groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'instance_post'],
                  as_index=False)['weight'].sum().sort_values(by='weight', ascending=False).copy()
# Normalize by total weights of each post-target
TuTuA2_in = normalize_weights_by_total(TuTuA2_in, group_col='instance_post')

TuTuA2_in_conn_mat = connection_table_to_matrix(TuTuA2_in,
                        weight_col=weight_type,
                        group_cols=['type_pre', 'instance_post'],
                        sort_by= ['weight', 'weight'])

if weight_type == 'percent_of_total':
    #TuTuA2_in_conn_mat[TuTuA2_in_conn_mat==0] = np.nan
    colorbar_label = 'percent of total inputs'
    vmin = 0
    vmax = 0.5
elif weight_type == 'log':
    TuTuA2_in_conn_mat = util.log_weights(TuTuA2_in_conn_mat)
    vmax = TuTuA2_in_conn_mat.max().max()
    vmin = TuTuA2_in_conn_mat.min().min()
    colorbar_label = 'log(weight)'
else:
    TuTuA2_in_conn_mat = TuTuA2_in_conn_mat
    colorbar_label = 'weight'
    vmin=None; vmax=None;
    
fig, ax = plt.subplots(figsize=(6, 6))
fig = plot_connection_matrix(TuTuA2_in_conn_mat, ax=ax, 
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
# TuTuA_2: plot inputs x outputs 
weight_label = 'percent_of_total'
TuTuA_in_, TuTuA2_out_, TuTuA2_in_out = matmul_conn_matrices(
                                TuTuA2_inputs_conn_df, TuTuA2_outputs_conn_df, 
                                 weight_label='percent_of_total',
                                 sort_rows='weight', sort_cols='weight',
                                 conn1_pre='type_pre', conn1_post='instance_post',
                                 conn2_pre='instance_pre', conn2_post='type_post',
                                 return_all=True)

fig, axn = plt.subplots(1, 3, figsize=(12, 4))
plot_connection_matrix(TuTuA_in_, ax=axn[0],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_label,
                       normalize_colors=True)
axn[0].set_title('TuTuA_2 inputs')
plot_connection_matrix(TuTuA2_out_, ax=axn[1],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_label,
                       normalize_colors=True)
axn[1].set_title('TuTuA_2 outputs')
plot_connection_matrix(TuTuA2_in_out, ax=axn[2],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_label,
                       normalize_colors=True)
axn[2].set_title('TuTuA_2 inputs X outputs')

#%%
# Total inputs and outputs for P1_1b
# ======================================================
P1_types = ['P1_1b', 'P1_1a']
P1_1_inputs_neuron_df, P1_1_inputs_conn_df = neu.fetch_adjacencies(
                                            sources=None,
                                            targets=NC(type=P1_types), 
                                            client=c, min_total_weight=10)
P1_1_inputs_conn_df = neu.merge_neuron_properties(P1_1_inputs_neuron_df, 
                                                  P1_1_inputs_conn_df, 
                                                  ['type', 'instance'])
#%
# P1: Group conn_df by type_pre, and sort by sum of weight
sorted_P1_1_inputs = P1_1_inputs_conn_df.groupby(['type_post', 
                                                    'type_pre', 
                                                    ])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
print('P1_1 inputs:')
print(sorted_P1_1_inputs.iloc[0:20])
#%
# Get all P1 outputs
P1_1_outputs_neuron_df, P1_1_outputs_conn_df = neu.fetch_adjacencies(
                                            sources=NC(type=P1_types), 
                                            targets=None,
                                            client=c, min_total_weight=10)
P1_1_outputs_conn_df = neu.merge_neuron_properties(P1_1_outputs_neuron_df, P1_1_outputs_conn_df, ['type', 'instance'])
#P1_1_outputs_conn_df['side'] = P1_1_outputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
#%
# P1_1: Group conn_df by type_post, and sort by sum of weight
sorted_P1_1_outputs = P1_1_outputs_conn_df.groupby(['type_post', 
                                                    'type_pre', 
                                                    ])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
sorted_P1_1b_outputs = sorted_P1_1_outputs[sorted_P1_1_outputs['type_pre']=='P1_1b']
print('P1_1b outputs:')
print(sorted_P1_1b_outputs.iloc[0:20])

#%%
# P1_1 total inputs
P1_1_inputs_aggr = P1_1_inputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     
total_P1_1_inputs = P1_1_inputs_aggr['weight'].sum()
P1_1_inputs_aggr['percent_of_total'] = P1_1_inputs_aggr['weight'] / total_P1_1_inputs

P1_1_inputs_by_type = P1_1_inputs_aggr.groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)

P1_1a_inputs_by_type = P1_1_inputs_aggr[P1_1_inputs_aggr['type_post']=='P1_1a'].groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)
#print(P1_1a_inputs_by_type)
P1_1b_inputs_by_type = P1_1_inputs_aggr[P1_1_inputs_aggr['type_post']=='P1_1b'].groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)
print("Top P1_1b inputs:")
print(P1_1b_inputs_by_type.iloc[0:20])

#%% P1_1 total outputs
P1_1_outputs_aggr = P1_1_outputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     
total_P1_1_outputs = P1_1_outputs_aggr['weight'].sum()
P1_1_outputs_aggr['percent_of_total'] = P1_1_outputs_aggr['weight'] / total_P1_1_outputs

P1_1b_outputs_by_type = P1_1_outputs_aggr[P1_1_outputs_aggr['type_pre']=='P1_1b']\
                                .groupby('type_post')['percent_of_total']\
                                .sum().sort_values(ascending=False)
print("Top P1_1b outputs:")
print(P1_1b_outputs_by_type.iloc[0:20])

#%%

# Plot connection matrix showing inputs to P1_1b as the rows,
# and outputs from P1_1b as the columns
# ------------------------------------------------------------
P1_1b_inputs_conn_df = P1_1_inputs_conn_df[P1_1_inputs_conn_df['type_post']=='P1_1b']
P1_1b_outputs_conn_df = P1_1_outputs_conn_df[P1_1_outputs_conn_df['type_pre']=='P1_1b']

# Normalize inputs by total inputs to P1_1b
P1_1b_inputs_conn_df = normalize_weights_by_total(P1_1b_inputs_conn_df, group_col='instance_post')

# Get all inputs to P1_1b outputs
inputs_to_P1_1b_outputs_neurons, inputs_to_P1_1b_outputs_conns = neu.fetch_adjacencies(
                                          sources=None,
                                          targets=NC(type=P1_1b_outputs_conn_df['type_post'].unique()),
                                          client=c, min_total_weight=10)
inputs_to_P1_1b_outputs_conns = neu.merge_neuron_properties(inputs_to_P1_1b_outputs_neurons, 
                                                    inputs_to_P1_1b_outputs_conns, ['type', 'instance'])
# Normalize P1_1b outputs by total outputs they get from ALL sources
inputs_to_P1_1b_outputs_conns = normalize_weights_by_total(inputs_to_P1_1b_outputs_conns, group_col='instance_post')

# Update output conn_df with normalized weights
P1_1b_outputs_conn_df = inputs_to_P1_1b_outputs_conns[inputs_to_P1_1b_outputs_conns['type_pre']=='P1_1b'].copy() #groupby('type_post')['weight'].sum()


#%%
# Combine connection matrices
# ------------------------------------------------------------
weight_label = 'percent_of_total';
# P1_1b_inputs_conn = connection_table_to_matrix(P1_1b_inputs_conn_df,
#                         group_cols=['type_pre', 'instance_post'],
#                         sort_by= ['weight', 'weight'],
#                         weight_col=weight_label)
# P1_1b_outputs_conn = connection_table_to_matrix(P1_1b_outputs_conn_df,
#                         group_cols=['instance_pre', 'type_post'],
#                         sort_by= ['weight', 'weight'],
#                         weight_col=weight_label)
# # sort labels
# intermediate_neurons = P1_1b_inputs_conn_df['instance_post'].unique()
# P1_1b_inputs_conn = sort_matrix_labels(P1_1b_inputs_conn, conn_df=P1_1b_inputs_conn_df, 
#                                        sort_rows='weight', sort_cols=intermediate_neurons)
# P1_1b_outputs_conn = sort_matrix_labels(P1_1b_outputs_conn, conn_df=P1_1b_outputs_conn_df, 
#                                        sort_rows=intermediate_neurons, sort_cols='weight')
# # Do matrix multiplication of inputs and outputs
# P1_in_out = P1_1b_inputs_conn.dot(P1_1b_outputs_conn)

P1_1b_in, P1_1b_out, P1_in_out = matmul_conn_matrices(P1_1b_inputs_conn_df, P1_1b_outputs_conn_df, 
                                 weight_label=weight_label,
                                 sort_rows='weight', sort_cols='weight',
                                 conn1_pre='type_pre', conn1_post='instance_post',
                                 conn2_pre='instance_pre', conn2_post='type_post',
                                 return_all=True)
#% PLOT
vmin = 0; vmax=0.1;
# Make a big grid of plots using GridSpec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2)
                      #width_ratios=[1, 1], height_ratios=[1, 1])
axn = [fig.add_subplot(gs[0, 0]), 
       fig.add_subplot(gs[0, 1]), 
       fig.add_subplot(gs[1:, 0:])] #, fig.add_subplot(gs[1, 1])]
plot_connection_matrix(P1_1b_in, ax=axn[0],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_label,
                       normalize_colors=True)
axn[0].set_title('P1_1b inputs (% of total inputs to P1_1b)')
plot_connection_matrix(P1_1b_out, ax=axn[1],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_label,
                       normalize_colors=True)
axn[1].set_title('P1_1b outputs (% total inputs to targets)')

axn[2].set_title('P1_1b inputs X outputs')
plot_connection_matrix(P1_in_out, ax=axn[2],
                       vmin=vmin, vmax=None,
                       colorbar_label=weight_label,
                       normalize_colors=True,
                       show_all_row_labels=True,
                       show_all_col_labels=True)
axn[2].set_title('P1_1b inputs X outputs')


#%%
# Plot connection matrix between all P1 and all LC
# ------------------------------------------------------------
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

#%%
# Get all P1 INPUTS:
P1_inputs_neuron_df, P1_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                               targets=NC(type='P1.*'))
P1_inputs_conn_df = neu.merge_neuron_properties(P1_inputs_neuron_df, P1_inputs_conn_df, ['type', 'instance'])
#P1_inputs_conn_df['side'] = P1_inputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)

# Group by type_pre and divide its weight onto a given type_post by dividing by the total weights onto that type_post
total_P1_inputs = P1_inputs_conn_df.groupby('type_post', as_index=False)['weight'].sum()
for type_post, df_ in P1_inputs_conn_df.groupby('type_post'):
    df_['percent_of_total'] = df_['weight'] / total_P1_inputs[total_P1_inputs['type_post']==type_post]['weight'].values[0]
    P1_inputs_conn_df.loc[P1_inputs_conn_df['type_post']==type_post, 'percent_of_total'] = df_['percent_of_total']
#%
# P1: Group conn_df by type_pre, and sort by sum of weight
sorted_P1_inputs = P1_inputs_conn_df.groupby(['type_post', 'type_pre'],
                                               as_index=False)['percent_of_total'].sum().sort_values(by='percent_of_total', ascending=False)
print('P1inputs:')
print(sorted_P1_inputs.iloc[0:20])
#%%
# Get ALL P1 OUTPUTS:
# ------------------------------------------------------------
P1_outputs_neuron_df, P1_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type='P1.*'), 
                                                                 targets=None)
P1_outputs_conn_df = neu.merge_neuron_properties(P1_outputs_neuron_df, P1_outputs_conn_df, ['type', 'instance'])

# Out of all the outputs a given P1 type makes, what percent goes to a given target
total_P1_outputs = P1_outputs_conn_df.groupby('type_pre', as_index=False)['weight'].sum()
for type_pre, df_ in P1_outputs_conn_df.groupby('type_pre'):
    df_['percent_of_total'] = df_['weight'] / total_P1_outputs[total_P1_outputs['type_pre']==type_pre]['weight'].values[0]
    P1_outputs_conn_df.loc[P1_outputs_conn_df['type_pre']==type_pre, 'percent_of_total'] = df_['percent_of_total']
#%
# P1_1: Group conn_df by type_post, and sort by sum of weight
sorted_P1_outputs = P1_outputs_conn_df.groupby(['type_pre', 'type_post'],
                                               as_index=False)['percent_of_total'].sum().sort_values(by='percent_of_total', ascending=False)
print('P1outputs:')
print(sorted_P1_outputs.iloc[0:20])

#%% 
# P1 INPUTS:  Plot input matrix
clear_empty_cells = True
topN = 50
min_input_weight = 0.01
P1_input_conn_matrix = connection_table_to_matrix(P1_inputs_conn_df,
                                    weight_col='percent_of_total',
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])

# Manually sort P1 labels (filter out None values)
pre_order = P1_input_conn_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
post_order = sorted([x if x is not None else 'None' for x in P1_inputs_conn_df['type_post'].unique() if x is not None], key=util.natsort)
P1_input_conn_matrix = P1_input_conn_matrix.reindex(index=pre_order, columns=post_order)

# Only take top N rows
P1_summed_inputs = P1_input_conn_matrix.sum(axis=1).sort_values(ascending=False)
P1_summed_inputs.iloc[0:30]

#%
# Only include connections with some min weight
if min_input_weight > 0:
    P1_input_conn_matrix[P1_input_conn_matrix < min_input_weight] = 0
# Drop any rows that have all NaN values
#P1_input_conn_filt = P1_input_conn_filt.dropna(axis=0, how='all')
#
P1_input_conn_filt = P1_input_conn_matrix.loc[P1_summed_inputs.index[0:topN]]
# Plot
if clear_empty_cells:
    P1_input_conn_filt[P1_input_conn_filt==0] = np.nan
    
vmin = min_input_weight
vmax = 0.2
colorbar_label = 'percent of total inputs'
# Plot P1 input matrix
#fig, ax = plt.subplots(figsize=(6, 15))
fig = plot_connection_matrix(P1_input_conn_filt, ax=None, #ax,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True, 
                       grid_color='k', grid_lw=0.01)
fig.axes[0].set_xlabel('Post-synaptic P1 type')
fig.axes[0].set_title('Top {} P1 inputs (min weight: {})'.format(topN, min_input_weight))


#%%
# Plot all P1 outputs

P1_output_conn_matrix = connection_table_to_matrix(P1_outputs_conn_df,
                                    weight_col='percent_of_total',
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])

# Manually sort P1 labels (filter out None values)
#pre_order = P1_output_conn_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
pre_order = sorted(P1_output_conn_matrix.index.unique(), key=util.natsort)
post_order = P1_output_conn_matrix.sum(axis=0).sort_values(ascending=False).index.tolist() #sorted([x if x is not None else 'None' for x in P1_outputs_conn_df['type_post'].unique() if x is not None], key=util.natsort)
P1_output_conn_matrix = P1_output_conn_matrix.reindex(columns=post_order,
                                                      index=pre_order)

#%
clear_empty_cells = False
topN = 50
min_output_weight = 0.05
vmax = None

# Only take top N outputs
# Summing across the rows should equal to 1, since normalized each PRE by its total outputs
# Sum across columns: Which targets of P1 types are getting the most?
P1_summed_outputs = P1_output_conn_matrix.sum(axis=0).sort_values(ascending=False)
P1_summed_outputs.iloc[0:30]
P1_output_conn_matrix_filt = P1_output_conn_matrix[P1_summed_outputs.index[0:N]].copy()

# Only include connections with some min weight
if min_output_weight > 0:
    P1_output_conn_matrix[P1_output_conn_matrix < min_output_weight] = 0    
if clear_empty_cells:
    P1_output_conn_matrix_filt[P1_output_conn_matrix_filt==0] = np.nan
    
# Plot P1 output matrix
fig = plot_connection_matrix(P1_output_conn_matrix_filt, ax=None, #ax,
                       vmin=vmin, vmax=None,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=False)
                       #grid_color='k', grid_lw=0.0)
fig.axes[0].set_xlabel('Post-synaptic P1 type')
fig.axes[0].set_title('Top {} P1 outputs (min weight: {})'.format(topN, min_output_weight))

# Highlight all outputs of P1_1b, where value greater than 0
P1_1b_outputs = P1_output_conn_matrix_filt.loc['P1_1b'].sort_values(ascending=False)
top_P1_1b_outputs = P1_1b_outputs.iloc[0:3]
highlight_row_or_column(fig.axes[0], P1_output_conn_matrix_filt, 
                        column_label=top_P1_1b_outputs.index.tolist(), 
                        color='red', linewidth=1)

#%%

# Biggest NON-LC10 input to TuTuA_2 is SMP054
# SMP054 gets most input from aIPG types


# Bigggest P1_1b output is to these SIP neurons-- where do they go?
top_SIP = ['SIP104m', 'SIP122m', 'SIP103m']
SIP_ouputs_neuron_df, SIP_ouputs_conn_df = neu.fetch_adjacencies(sources=NC(type=top_SIP),
                                                                 targets=None)
SIP_ouputs_conn_df = neu.merge_neuron_properties(SIP_ouputs_neuron_df, 
                                                 SIP_ouputs_conn_df, ['type', 'instance'])
SIP_ouputs_aggr = SIP_ouputs_conn_df.groupby(['bodyId_pre', 
                                              'bodyId_post', 
                                              'type_pre', 'type_post'],
                                            as_index=False)['weight'].sum().sort_values(by='weight', 
                                                ascending=False)     
total_SIP_ouputs = SIP_ouputs_aggr['weight'].sum()
SIP_ouputs_aggr['percent_of_total'] = SIP_ouputs_aggr['weight'] / total_SIP_ouputs

top_SIP_outputs = SIP_ouputs_aggr[SIP_ouputs_aggr['type_pre'].isin(top_SIP)].groupby('type_post')['percent_of_total'].sum().sort_values(ascending=False)
print(top_SIP_outputs.iloc[0:20])


#%%

#%%

# Plot P1 to P1 connections
P1_P1_neuron_df, P1_P1_conn_df = neu.fetch_adjacencies(sources=NC(type='P1.*'),
                                                       targets=NC(type='P1.*'))
P1_P1_conn_df = neu.merge_neuron_properties(P1_P1_neuron_df, P1_P1_conn_df, ['type', 'instance'])
#%
# P1_1: Group conn_df by type_post, and sort by sum of weight
sorted_P1_P1 = P1_P1_conn_df.groupby(['type_post', 'type_pre'],
                                               as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)
print('P1_P1:')
print(sorted_P1_P1.iloc[0:20])

# Normalize input weights by target inputs
total_P1_P1_inputs = P1_P1_conn_df.groupby('type_post', as_index=False)['weight'].sum()
for type_post, df_ in P1_P1_conn_df.groupby('type_post'):
    df_['percent_of_total'] = df_['weight'] / total_P1_P1_inputs[total_P1_P1_inputs['type_post']==type_post]['weight'].values[0]
    P1_P1_conn_df.loc[P1_P1_conn_df['type_post']==type_post, 'percent_of_total'] = df_['percent_of_total']
#%
# P1_P1: Group conn_df by type_pre, and sort by sum of weight
sorted_P1_P1 = P1_P1_conn_df.groupby(['type_post', 'type_pre'],
                                               as_index=False)['percent_of_total'].sum().sort_values(by='percent_of_total', ascending=False)
print('P1_P1 inputs normalized by target inputs:')
print(sorted_P1_P1.iloc[0:20])

# Create connection matrix
P1_P1_conn_matrix = connection_table_to_matrix(P1_P1_conn_df,
                                    weight_col='percent_of_total',
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])
# Sort labels alphabetically
pre_order = sorted(P1_P1_conn_matrix.index.unique(), key=util.natsort)
post_order = sorted(P1_P1_conn_matrix.columns.unique(), key=util.natsort)
P1_P1_conn_matrix = P1_P1_conn_matrix.reindex(index=pre_order, columns=post_order)
#%
# Plot P1_P1 connection matrix
vmin=None;vmax=None;
fig, ax = plt.subplots(figsize=(6, 6))
plot_connection_matrix(P1_P1_conn_matrix, ax=ax,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True, 
                       grid_color=[0.8]*3, grid_lw=0.001)
ax.set_title('P1_P1 connections')
ax.set_xlabel('Post-synaptic P1 type')
ax.set_ylabel('Pre-synaptic P1 type')

#%%
# Cluster P1_P1_conn_matrix using cosine similarity
P1_P1_clustered, row_linkage, col_linkage = cluster_matrix_cosine_similarity(P1_P1_conn_matrix, method='ward')

# Plot clustered matrix
fig = plot_connection_matrix(P1_P1_clustered,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=False, 
                       grid_color=[0.8]*3, grid_lw=0.001)
ax = fig.axes[0]
ax.set_title('P1_P1 connections (cosine similarity clustered)')
ax.set_xlabel('Post-synaptic P1 type')
ax.set_ylabel('Pre-synaptic P1 type')

#%%
# Plot dendrograms to understand the clustering
fig_dendro = plot_dendrograms(row_linkage, col_linkage, 
                             row_labels=P1_P1_clustered.index.tolist(),
                             col_labels=P1_P1_clustered.columns.tolist())


#%%
# Test thresholded clustering (focuses on strong connections)
print("Testing thresholded clustering...")

# Method 1: Percentile-based thresholding (keep top 85% of connections per row/column)
P1_P1_clustered_thresh, row_linkage_thresh, col_linkage_thresh, matrix_thresh = \
    cluster_matrix_cosine_similarity_thresholded(P1_P1_conn_matrix, 
                                                 method='ward', 
                                               threshold_percentile=90)

# Plot comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Original clustering
plot_connection_matrix(P1_P1_clustered, ax=ax1, normalize_colors=True, show_grid=False)
ax1.set_title('Original Clustering (All Connections)')

# Thresholded clustering
plot_connection_matrix(P1_P1_clustered_thresh, ax=ax2, normalize_colors=True, show_grid=False)
ax2.set_title('Thresholded Clustering (Top 85% per row/column)')

# Show thresholded matrix
plot_connection_matrix(matrix_thresh, ax=ax3, normalize_colors=True, show_grid=False)
ax3.set_title('Thresholded Matrix (Used for Clustering)')

# Show difference
diff_matrix = P1_P1_conn_matrix - matrix_thresh
plot_connection_matrix(diff_matrix, ax=ax4, normalize_colors=True, show_grid=False)
ax4.set_title('Removed Connections (Original - Thresholded)')

plt.tight_layout()

#%%
# Plot dendrograms for thresholded clustering
fig_dendro_thresh = plot_dendrograms(row_linkage_thresh, col_linkage_thresh, 
                                   row_labels=P1_P1_clustered_thresh.index.tolist(),
                                   col_labels=P1_P1_clustered_thresh.columns.tolist())
fig_dendro_thresh.suptitle('Thresholded Clustering Dendrograms', fontsize=16)

#%%
# Create comprehensive cluster analysis
thresholded=False
if thresholded:
    fig_analysis, row_clusters, col_clusters = plot_cluster_analysis(
        P1_P1_clustered_thresh, row_linkage_thresh, col_linkage_thresh, 
        n_clusters=5, figsize=(20, 14), grid_lw=0, 
        show_all_labels=True, label_fontsize=8)
else:
    fig_analysis, row_clusters, col_clusters = plot_cluster_analysis(
        P1_P1_clustered, row_linkage, col_linkage, 
        n_clusters=5, figsize=(20, 14), grid_lw=0, 
        show_all_labels=True, label_fontsize=8)
#%%


#%%
# No side or ROI:
LC10a_inputs_aggr = LC10a_inputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                             as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)    
total_LC10a_inputs = LC10a_inputs_aggr['weight'].sum()
LC10a_inputs_aggr['percent_of_total'] = LC10a_inputs_aggr['weight'] / total_LC10a_inputs

LC10a_inputs_by_type = LC10a_inputs_aggr.groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)
print(LC10a_inputs_by_type)

LC10a_outputs_aggr = LC10a_outputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     

# TuTuA_2 total inputs
TuTuA2_inputs_aggr = TuTuA2_inputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     
total_TuTuA2_inputs = TuTuA2_inputs_aggr['weight'].sum()
TuTuA2_inputs_aggr['percent_of_total'] = TuTuA2_inputs_aggr['weight'] / total_TuTuA2_inputs

TuTuA2_inputs_by_type = TuTuA2_inputs_aggr.groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)
print(TuTuA2_inputs_by_type)
#%%

# Get SMP054 inputs and outputs:
SMP054_inputs_neuron_df, SMP054_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                               targets=NC(type='SMP054.*'))
SMP054_inputs_conn_df = neu.merge_neuron_properties(SMP054_inputs_neuron_df, SMP054_inputs_conn_df, ['type', 'instance'])
#%
SMP054_inputs_aggr = SMP054_inputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     
total_SMP054_inputs = SMP054_inputs_aggr['weight'].sum()
SMP054_inputs_aggr['percent_of_total'] = SMP054_inputs_aggr['weight'] / total_SMP054_inputs

SMP054_inputs_by_type = SMP054_inputs_aggr.groupby('type_pre')['percent_of_total'].sum().sort_values(ascending=False)
print(SMP054_inputs_by_type.iloc[0:20])

#%%
# SMP outputs
SMP054_outputs_neuron_df, SMP054_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type='SMP054.*'),
                                                               targets=None)
SMP054_outputs_conn_df = neu.merge_neuron_properties(SMP054_outputs_neuron_df, SMP054_outputs_conn_df, ['type', 'instance'])
#%
SMP054_outputs_aggr = SMP054_outputs_conn_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post'],    
                                                    as_index=False)['weight'].sum().sort_values(by='weight', 
                                                                         ascending=False)     
total_SMP054_outputs = SMP054_outputs_aggr['weight'].sum()
SMP054_outputs_aggr['percent_of_total'] = SMP054_outputs_aggr['weight'] / total_SMP054_outputs

SMP054_outputs_by_type = SMP054_outputs_aggr.groupby('type_post')['percent_of_total'].sum().sort_values(ascending=False)
print(SMP054_outputs_by_type.iloc[0:20])

#%%