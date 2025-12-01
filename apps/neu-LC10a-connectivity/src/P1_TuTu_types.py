'''
 # @ Author: Juliana Rhee
 # @ Filename: P1_TuTu_types.py
 # @ Create Time: 2025-10-13 10:53:42
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-10-13 10:53:48
 # @ Description: Plot the connectivity of P1 TuTu types to LC10a neurons
 '''

#%%
from operator import truediv
import os
import glob
import numpy as np
from numpy.ma.core import true_divide
import pandas as pd
import matplotlib.pyplot as plt
#from pandas.compat import F
import seaborn as sns

import neuprint as neu
from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import utils as util
import plotting as putil
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
import neuprint_funcs as npf
#%%

def sum_connection_matrix(conn_matrix, axis='rows'):
    '''
    Sum across rows or columns of a connection matrix.
    
    Args:
        conn_matrix: DataFrame or numpy array, the connection matrix to sum
        axis: str, 'rows' to sum across rows (axis=1), 'columns' to sum across columns (axis=0)
        
    Returns:
        pandas Series or numpy array with summed values for each row or column
    '''
    if axis == 'rows':
        return conn_matrix.sum(axis=1)
    elif axis == 'columns':
        return conn_matrix.sum(axis=0)
    else:
        raise ValueError("axis must be 'rows' or 'columns'")

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
dataset = 'male-cns:v0.9'
c = npf.get_neuprint_client(dataset=dataset)
version = c.fetch_version()
figid = f'{dataset}_{version}'
print(figid)

#%% Plot style
plot_style = 'dark'
min_fontsize=12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%% Output dir
rootdir = '/Volumes/Juliana/connectome'
output_dir = os.path.join(rootdir, 'analyses', 'neuprint', 'P1_TuTu_types')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Output directory: {output_dir}')

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
weight_var = 'percent_of_total'
groupby_cols=['roi_noside', 'type_pre', 'type_post', 'bodyId_pre', 'bodyId_post',
              'instance_pre', 'instance_post']
# bodyId_pre are INPUTS to targets, bodyId_post are the target IDs
LC10a_inputs_conn_df, LC10a_inputs_matrix = npf.get_conn_all_inputs(target_type='LC10a',
                                                        client=c, 
                                                        return_both=True, 
                                                        weight_var=weight_var,
                                                        min_total_weight=min_total_weight,
                                                        groupby_cols=groupby_cols)
LC10a_inputs_conn_df.groupby('type_post')['percent_of_total'].sum()
#%%
sorted_LC10a_inputs = LC10a_inputs_conn_df.sort_values(by='weight', ascending=False)
print(sorted_LC10a_inputs.iloc[0:20])
# #%%
# LC10a_inputs_neuron_df, LC10a_inputs_conn_df = neu.fetch_adjacencies(targets=NC(type=['LC10a']),
#                                                 min_total_weight=min_total_weight)
# LC10a_inputs_grouped= npf.merge_properties_and_group(LC10a_inputs_neuron_df, 
#                                                     LC10a_inputs_conn_df,
#                                                     groupby_cols=['roi_noside', 'type_pre'])
# sorted_LC10a_inputs = LC10a_inputs_grouped.sort_values(by='weight', ascending=False)
# 
# #LC10a_inputs_conn_df = neu.merge_neuron_properties(LC10a_inputs_neuron_df, LC10a_inputs_conn_df, ['type', 'instance'])
# # Extract side from roi
# #LC10a_inputs_conn_df = npf.extract_side_from_conn_df(LC10a_inputs_conn_df)
# #LC10a_inputs_conn_df['side'] = LC10a_inputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
# # Remove side from roi
# #LC10a_inputs_conn_df['roi_noside'] = LC10a_inputs_conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
# 
# #%
# # LC10a: Group conn_df by type_pre, and sort by sum of weight
# #sorted_LC10a_inputs = LC10a_inputs_conn_df.groupby(['roi_noside', 
# #                                                    'type_pre'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
# print('LC10a inputs:')
# print(sorted_LC10a_inputs.iloc[0:20])
 
#%%
# LC10a: Get all outputs: 
# -------------------------------------------------------
min_total_weight = 10
LC10a_outputs_conn_df, LC10a_outputs_matrix = npf.get_conn_all_outputs(source_type='LC10a',
                                                        client=c, 
                                                        return_both=True, 
                                                        weight_var=weight_var,
                                                        min_total_weight=min_total_weight,
                                                        norm_by_all_other_inputs=True,
                                                        groupby_cols=groupby_cols)
print(LC10a_outputs_conn_df.groupby('type_post')['percent_of_total'].sum())
sorted_LC10a_outputs = LC10a_outputs_conn_df.sort_values(by='weight', ascending=False)
print(sorted_LC10a_outputs.iloc[0:20])
#%%
# # Check output normalization 
# neuron_df, conn_df = neu.fetch_adjacencies(sources=None,
#                                                 targets='AOTU019',
#                                                 client=c,
#                                                 min_total_weight=min_total_weight)
# conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
# conn_df = npf.extract_side_from_conn_df(conn_df)
# grouped_conn_df = conn_df.groupby(['type_pre', 'type_post'], as_index=False)['weight'].sum()
# print(grouped_conn_df.sort_values(by='weight', ascending=False))
# 
# #%%
# grouped_conn_df = npf.norm_by_specified_inputs(grouped_conn_df, group_col='type_post')
# print(grouped_conn_df.sort_values(by='weight', ascending=False))
# 
# print(grouped_conn_df.sort_values(by='percent_of_total', ascending=False))
# 
# # compare:
# # AOTU019 is lower on list, some types show greater percent of inputs from LC10a
# LC10a_outputs_conn_df.groupby('type_post')['percent_of_total'].sum().sort_values(ascending=False).iloc[0:30]
# 
# # total weights to AOTU019 from LC10a are same
# LC10a_outputs_conn_df[LC10a_outputs_conn_df['type_post']=='AOTU019']['percent_of_total'].sum()
# 
 

#%%
# bodyId_pre are the LC10a source IDs, bodyId_post are the target IDs
# LC10a_outputs_neuron_df, LC10a_outputs_conn_df = neu.fetch_adjacencies(
#                                                         sources=NC(type=['LC10a']), 
#                                                         targets=None,
#                                                         min_total_weight=min_total_weight)
# LC10a_outputs_conn_df = neu.merge_neuron_properties(LC10a_outputs_neuron_df, LC10a_outputs_conn_df, ['type', 'instance'])
# LC10a_outputs_conn_df['side'] = LC10a_outputs_conn_df['roi'].str.extract(r'\(([LR])\)', expand=False)
# # Remove side from roi
# LC10a_outputs_conn_df['roi_noside'] = LC10a_outputs_conn_df['roi'].str.extract(r'^(.*)\(.*\)', expand=False)
# # Sort 
# sorted_LC10a_outputs = LC10a_outputs_conn_df.groupby(['roi_noside', 
#                                                       'type_post'])['weight'].sum().reset_index().sort_values(by='weight', ascending=False)
# print('LC10a outputs:')
# print(sorted_LC10a_outputs.iloc[0:20])
# 

#%%
 # Add any missing columns
def group_conn_df(conn_df, pre_variable='type_pre', post_variable='type_post', 
                  pre_grouper='roi_noside', post_grouper='roi_noside', 
                  weight_type='weight', 
                  group_cols=['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post']):
    '''
    Group a connection dataframe by a given variable and sort by the weights.
    Args:
        conn_df: DataFrame, the connection dataframe
        pre_variable: str, the column to group by for the rows
        post_variable: str, the column to group by for the columns
        pre_grouper: str, the column to sort the rows by
        post_grouper: str, the column to sort the columns by
        weight_type: str, the column to use for the weights
    Returns:
        conn_df: DataFrame, the grouped connection dataframe
    '''
    
    #group_cols = ['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post']
    for col in [pre_variable, post_variable, pre_grouper, post_grouper]:
        if col not in group_cols:
            group_cols.append(col)
        
    # Get grouped connections df
    conn_ = conn_df.groupby(group_cols,
                as_index=False)[weight_type].sum().sort_values(by=weight_type, ascending=False)
    
    return conn_
                         
#%%
# Show Connection Matrix for LC10a INPUTS
separate_by_side = True
sort_by_weights_only = False
#weight_type = 'percent_of_total'
weight_type = 'weight'
use_log_weights = weight_type == 'weight'
vmax = 0.001 if weight_type == 'percent_of_total' else None

post_variable = 'bodyId_post'
if separate_by_side:
    pre_variable = 'instance_pre'
    pre_grouper = 'side_pre'
    post_grouper = 'side_post'
    highlight_rows = ['TuTuA_2_L', 'TuTuA_2_R']
elif sort_by_weights_only:
    pre_variable = 'type_pre'
    pre_grouper = weight_type
    post_grouper = weight_type
    highlight_rows = ['TuTuA_2']
else:
    pre_variable = 'type_pre'
    pre_grouper = 'roi_noside'
    post_grouper = 'roi_noside'
    highlight_rows = ['TuTuA_2']
    
# Group conn df
LC10a_in = group_conn_df(LC10a_inputs_conn_df, pre_variable=pre_variable, post_variable=post_variable, 
                         pre_grouper=pre_grouper, post_grouper=post_grouper, weight_type=weight_type)

# Connection mat
LC10a_in_conn_matrix = connection_table_to_matrix(LC10a_in,
                        weight_col=weight_type,
                        group_cols=[pre_variable, post_variable],
                        sort_by= [ pre_grouper, post_grouper])
print(LC10a_in_conn_matrix.shape)
#%
# sort index
LC10a_in_conn_matrix = npf.sort_matrix_labels(LC10a_in_conn_matrix, 
                                          conn_df=LC10a_in,  
                                          sort_rows_by=pre_grouper,
                                          sort_cols_by=post_grouper,
                                          sorted_var_name=None,
                                          weight_var=weight_type)
#%
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
    colorbar_label = f'log({weight_type})'
else:
    LC10a_conn = LC10a_in_conn_matrix
    colorbar_label = weight_type.replace('_', ' ')
fig = npf.plot_grouped_connection_matrix(LC10a_conn, LC10a_in, figsize=(20,12),
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
                                     show_all_col_labels=False,
                                     colorbar_label=colorbar_label,
                                     normalize_colors=True, vmax=vmax)
fig.axes[0].set_title(f'LC10a inputs (min. weight = {min_total_weight})')

npf.highlight_row_or_column(fig.axes[0], LC10a_conn, row_label=highlight_rows,
                        color='k', linewidth=2)


putil.label_figure(fig, figid)
figname = f'LC10a_inputs_{weight_type}_{min_total_weight}'
print(figname)

plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)

#%%
# Bar chart of strongest inputs by type and ROI
def barplot_top_n(conn_df, groupby='type_pre', 
                    weight_type='percent_of_total', n_top=20,
                    use_log_weights=False):
    #use_log_weights = weight_type == 'weight'
    sorted_inputs = conn_df.groupby([groupby], as_index=False)\
                            [weight_type].sum()\
                            .sort_values(by=weight_type, ascending=False)
    if use_log_weights:
        sorted_inputs['log_weight'] = np.log(sorted_inputs[weight_type])
        weight_var = 'log_weight'
    else:
        weight_var = weight_type
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=groupby, y=weight_var, ax=ax,
                data=sorted_inputs.iloc[0:n_top], color=[0.7]*3)
    # Make x-tick labels vertical 
    ax.tick_params(axis='x', labelrotation=90)
    # Only plot a subset of the x-tick labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xlabel(groupby)
    ax.set_ylabel(weight_var.replace('_', ' '))
    plt.subplots_adjust(bottom=0.2)

    return fig, ax
#%%
# BARPLOT: LC10a inputs
weight_type = 'percent_of_total'
#weight_type = 'weight'
n_top = 20
fig, ax = barplot_top_n(LC10a_inputs_conn_df, 
                        groupby='type_pre', 
                        weight_type=weight_type, n_top=n_top,
                        use_log_weights=False)
ax.set_title(f'Strongest inputs by type (min. weight = {min_total_weight})')
plt.subplots_adjust(bottom=0.2)
#plt.show()
putil.label_figure(fig, figid)
figname = f'LC10a_top{n_top}_inputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%
# LC10a OUTPUTS: 
weight_type = 'weight'
use_log_weights = weight_type == 'weight'
vmax = 0.001 if weight_type == 'percent_of_total' else None

pre_variable = 'bodyId_pre'
post_variable = 'type_post'

sort_by_weights_only=False

if sort_by_weights_only:
    pre_grouper = weight_type
    post_grouper = weight_type
else:
    pre_grouper = 'roi_noside'
    post_grouper = 'roi_noside'
# Get conn matrix       
LC10a_out_df = group_conn_df(LC10a_outputs_conn_df, pre_variable=pre_variable, post_variable=post_variable, 
                    pre_grouper=pre_grouper, post_grouper=post_grouper, weight_type=weight_type,
                    group_cols=['bodyId_pre', 'type_pre', 'type_post', 'roi_noside'])

LC10a_out_conn_matrix = connection_table_to_matrix(LC10a_out_df,
                                weight_col=weight_type,
                                group_cols=['bodyId_pre', 'type_post'],
                                sort_by= [weight_type, post_grouper])#'weight']) #, 'bodyId', sort_by='instance')    
# Sort index
LC10a_out_conn_matrix = npf.sort_matrix_labels(LC10a_out_conn_matrix, 
                                          conn_df=LC10a_out_df,  
                                          sort_rows_by=pre_grouper,
                                          sort_cols_by=post_grouper,
                                          sorted_var_name=None,
                                          weight_var=weight_type)

# Colormap
post_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(LC10a_out_df[post_grouper].unique())}

# Plot LC10a outputs - annotate columns (post/outputs)
if use_log_weights:
    LC10a_out_mat = util.log_weights(LC10a_out_conn_matrix)
    colorbar_label = f'log({weight_type})'
else:
    LC10a_out_mat = LC10a_out_conn_matrix
    colorbar_label = weight_type.replace('_', ' ')
fig = npf.plot_grouped_connection_matrix(LC10a_out_mat, 
                                     LC10a_out_df, 
                                     post_grouper_dict=post_grouper_dict,
                                     sorted_by_grouper=sort_by_weights_only==False,
                                     pre_grouper = pre_grouper,
                                     post_grouper = post_grouper,
                                     pre_variable=pre_variable,
                                     post_variable=post_variable,
                                     annotate_rows=True,
                                     annotate_cols=True, 
                                     show_all_col_labels=True,
                                     colorbar_label=colorbar_label, 
                                     vmax=vmax) #None)
fig.axes[0].set_title('LC10a outputs')
#%
# Add a thin box around a specified column or row based on the label
npf.highlight_row_or_column(fig.axes[0], LC10a_out_mat, 
                        column_label=['AOTU019', 'AOTU025', 'P1_1b'],
                        color='k', linewidth=2)

# save
putil.label_figure(fig, figid)
figname = f'LC10a_outputs_{weight_type}_{min_total_weight}'
print(figname)
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)

#%%
# BARPLOT: LC10a outputs
weight_type = 'percent_of_total'
#weight_type = 'weight'
n_top = 20
fig, ax = barplot_top_n(LC10a_outputs_conn_df, groupby='type_post', 
                        weight_type=weight_type, n_top=n_top,
                        use_log_weights=False)
ax.set_title(f'Strongest outputs by type (min. weight = {min_total_weight})')
plt.subplots_adjust(bottom=0.2)
putil.label_figure(fig, figid)


figname = f'LC10a_top_{n_top}_outputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)
#%%
# Get percent of total weights for IN
#LC10a_inputs_conn_df = npf.norm_by_specified_inputs(LC10a_inputs_conn_df, group_col='bodyId_post')
#LC10a_outputs_conn_df = npf.get_and_norm_by_total_inputs(LC10a_outputs_conn_df, 
                                                     normalize_group_col='type_post')

#%%
# LC10a:  Plot IN x OUT
plot_inputs = True
vmin = 0
vmax = 0.1
LC10a_in_, LC10a_out_, LC10a_in_out = npf.matmul_conn_matrices(
                                LC10a_inputs_conn_df, LC10a_outputs_conn_df, 
                                 weight_label='percent_of_total',
                                 sort_rows='weight', sort_cols='weight',
                                 conn1_pre='type_pre', conn1_post='bodyId_post',
                                 conn2_pre='bodyId_pre', conn2_post='type_post',
                                 return_all=True)
if plot_inputs:
    fig, ax = plt.subplots(figsize=(25, 10))
    npf.plot_connection_matrix(LC10a_in_, ax=ax,  
                        show_all_row_labels=True, show_all_col_labels=True,
                        vmin=vmin, vmax=vmax,
                        colorbar_label='% total inputs')
    ax.set_title('LC10a inputs')

    fig, ax = plt.subplots(figsize=(15, 25))
    npf.plot_connection_matrix(LC10a_out_, ax=ax, 
                        show_all_row_labels=True, show_all_col_labels=True, 
                        vmin=vmin, vmax=vmax,
                        colorbar_label='% total outputs')
    ax.set_title('LC10a outputs')

fig, ax = plt.subplots(figsize=(20, 10))
npf.plot_connection_matrix(LC10a_in_out, ax=ax, 
                       vmin=vmin, vmax=vmax,
                       show_all_row_labels=True,
                       show_all_col_labels=True, 
                       colorbar_label='% total inputs * outputs')
ax.set_title('LC10a inputs * outputs')
plt.show()


# %%
# Get all TuTuA_2 neurons
# ======================================================
TuTuA2_neurons, TuTuA2_roi_counts = neu.fetch_neurons(NC(type='TuTuA_2', 
                                                         client=c))
#%%
# Get all inputs
TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df = neu.fetch_adjacencies(
                                                                sources=None,
                                                                targets=NC(type=['TuTuA_2']),
                                                                min_total_weight=10)
TuTuA2_inputs_conn_df = neu.merge_neuron_properties(TuTuA2_inputs_neuron_df, TuTuA2_inputs_conn_df, ['type', 'instance'])

# Extract side info
TuTuA2_inputs_conn_df = npf.extract_side_from_conn_df(TuTuA2_inputs_conn_df)
# Add percent of total weight: group_col should result in all total weights for that group 
# summing to 1. To make it sum to 1 by type_pre, group by type_post?
TuTuA2_inputs_conn_df = npf.norm_by_specified_inputs(TuTuA2_inputs_conn_df, 
                                                   group_col='instance_post')

# TuTuA_2: Group conn_df by type_pre, and sort by sum of weight
sorted_TuTuA2_inputs = TuTuA2_inputs_conn_df.groupby(['roi', 'type_pre', 'side_pre', 'side_post'], \
                                            as_index=False)['percent_of_total'].sum()\
                                            .sort_values(by=['percent_of_total', 'percent_of_total'], 
                                            ascending=False)
print('TuTuA_2 inputs:')
print(sorted_TuTuA2_inputs.iloc[0:20])

#%%
# Get all outputs
TuTuA2_outputs_neuron_df, TuTuA2_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type=['TuTuA_2']), 
                                                                         targets=None,
                                                                         min_total_weight=10)
TuTuA2_outputs_conn_df = neu.merge_neuron_properties(TuTuA2_outputs_neuron_df, TuTuA2_outputs_conn_df, ['type', 'instance'])
# Extract side info
TuTuA2_outputs_conn_df0 = npf.extract_side_from_conn_df(TuTuA2_outputs_conn_df)
# Add percent of total weight
TuTuA2_outputs_conn_df = npf.get_and_norm_by_total_inputs(TuTuA2_outputs_conn_df0, c, 
                                                    normalize_group_col='type_post',
                                                    groupby_cols=['type_pre', 'type_post', 'roi_noside', 
                                                    'instance_post', 'instance_pre'])

#%%
# TuTuA_2 inputs: Aggregate all weights (aggregate across ROIs) to get total connection weights
# ------------------------------------------------------------
weight_type = 'percent_of_total' # can be: 'weight', 'percent', 'log'

TuTuA2_in = TuTuA2_inputs_conn_df\
                  .groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'instance_post'],
                  as_index=False)['percent_of_total'].sum().sort_values(by='percent_of_total', ascending=False).copy()
# Make conn mat
TuTuA2_in_conn_mat = connection_table_to_matrix(TuTuA2_in,
                        weight_col=weight_type,
                        group_cols=['type_pre', 'instance_post'],
                        sort_by= [weight_type, weight_type])

if weight_type == 'percent_of_total':
    #TuTuA2_in_conn_mat[TuTuA2_in_conn_mat==0] = np.nan
    colorbar_label = 'percent of total inputs'
    vmin = 0
    vmax = 0.4
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
fig = npf.plot_connection_matrix(TuTuA2_in_conn_mat, ax=ax, 
                             vmin=vmin, vmax=vmax,
                             show_all_row_labels=True,
                             show_all_col_labels=True,
                             colorbar_label=colorbar_label,
                             normalize_colors=True)
ax.set_title('TuTuA_2 inputs')
ax.set_xlabel('Post-synaptic bodyId')
ax.set_ylabel('Pre-synaptic bodyId')
#plt.show()

putil.label_figure(fig, figid)
figname = f'TuTuA2_inputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)
#
#%% 
# Plot TuTuA_2 inputs: Separate by ROI/side
# ------------------------------------------------------------
sort_weights = False #True
plot_by_side = True
weight_type = 'weight' # 'percent_of_total'
use_log_weights = weight_type == 'weight'
vmin = 0
vmax = 0.4
if plot_by_side:
    pre_variable = 'instance_pre'
    pre_grouper = 'side_pre' 
    sorted_by_grouper = True 
    sort_weights = False
    post_grouper = 'side_post'
else:
    pre_variable = 'type_pre'    
    pre_grouper = 'roi_noside'
    sorted_by_grouper = sort_weights is False
    post_grouper = 'roi_noside'
post_variable = 'instance_post'
manual_groups = sorted_by_grouper #False

# --------
TuTuA2_in_conn_matrix = connection_table_to_matrix(TuTuA2_inputs_conn_df,
                        weight_col=weight_type,
                        group_cols=[pre_variable, post_variable],
                        sort_by= [ pre_grouper, post_grouper]) #'weight']) 

if manual_groups: #sorted_by_grouper:
    in_vals = TuTuA2_in_conn_matrix.index.tolist()
    sort_by_roi = TuTuA2_inputs_conn_df[[pre_variable, pre_grouper]]\
                            .drop_duplicates()\
                            .sort_values(by=pre_grouper)
    TuTuA2_in_conn_matrix = TuTuA2_in_conn_matrix.loc[sort_by_roi[pre_variable].values]
    group_per_row = sort_by_roi[pre_grouper].values
else:
    group_per_row = None

    # Sort by weight
    sorted_TuTuA2_inputs = TuTuA2_in.groupby([ pre_variable, pre_grouper])[weight_type].sum().reset_index().sort_values(by=[weight_type], ascending=False)
    sorted_TuTuA2_inputs = TuTuA2_in.groupby(['type_pre'])[weight_type].sum().reset_index().sort_values(by=[weight_type], ascending=False)
    TuTuA2_in_conn_matrix = TuTuA2_in_conn_matrix.loc[sorted_TuTuA2_inputs['type_pre'].values]

# ROI colors
pre_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(TuTuA2_inputs_conn_df[pre_grouper].unique())}
post_grouper_dict = {roi: sns.color_palette("tab10")[i] 
                    for i, roi in enumerate(TuTuA2_inputs_conn_df[post_grouper].unique())}

#% Plot TuTuA_2 inputs
if use_log_weights:
    plot_TuTuA2_in = util.log_weights(TuTuA2_in_conn_matrix)
    colorbar_label = f'log({weight_type})'
    vmax = None
else:
    plot_TuTuA2_in = TuTuA2_in_conn_matrix
    colorbar_label = weight_type.replace('_', ' ') 
    vmax = 0.25

fig = npf.plot_grouped_connection_matrix(plot_TuTuA2_in, TuTuA2_inputs_conn_df, 
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
                                     colorbar_label=colorbar_label,
                                     vmax=vmax)
fig.axes[0].set_title('TuTuA_2 inputs')

# Highlight
npf.highlight_row_or_column(fig.axes[0], plot_TuTuA2_in, 
                        row_label=['SMP054_L', 'SMP054_R', 'LC10a_L', 'LC10a_R'],
                        color='k', linewidth=2, highlight_box=False)

putil.label_figure(fig, figid)
figname = f'TuTuA2_inputs_byside_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)


#%%
# TuTuA_2: plot inputs x outputs 
weight_type = 'percent_of_total'
TuTuA_in_, TuTuA2_out_, TuTuA2_in_out = npf.matmul_conn_matrices(
                                TuTuA2_inputs_conn_df, TuTuA2_outputs_conn_df, 
                                 weight_label=weight_type,
                                 sort_rows='weight', sort_cols='weight',
                                 conn1_pre='type_pre', conn1_post='instance_post',
                                 conn2_pre='instance_pre', conn2_post='type_post',
                                 return_all=True)
#%
vmax=0.2
fig, axn = plt.subplots(1, 3, figsize=(12, 6))
npf.plot_connection_matrix(TuTuA_in_, ax=axn[0],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_type,
                       normalize_colors=True,
                       show_all_row_labels=True)
axn[0].set_title('TuTuA_2 inputs')
npf.plot_connection_matrix(TuTuA2_out_, ax=axn[1],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_type,
                       normalize_colors=True,
                       show_all_col_labels=True)
axn[1].set_title('TuTuA_2 outputs')
npf.plot_connection_matrix(TuTuA2_in_out, ax=axn[2],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_type,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True)
axn[2].set_title('TuTuA_2 inputs X outputs')
plt.subplots_adjust(wspace=0.5, bottom=0.3)

putil.label_figure(fig, figid)
figname = f'TuTuA2_inputs_x_outputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

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
print("Top P1_1a/b outputs:")
print(P1_1b_outputs_by_type.iloc[0:20])

#%%

# Plot connection matrix showing inputs to P1_1b as the rows,
# and outputs from P1_1b as the columns
# ------------------------------------------------------------
#P1_1b_inputs_conn_df = P1_1_inputs_conn_df[P1_1_inputs_conn_df['type_post']=='P1_1b']
#P1_1b_outputs_conn_df = P1_1_outputs_conn_df[P1_1_outputs_conn_df['type_pre']=='P1_1b']

# Normalize inputs by total inputs to P1_1b
P1_1_inputs_conn_df = npf.norm_by_specified_inputs(P1_1_inputs_conn_df, group_col='instance_post')

# Get all inputs to P1_1b outputs
P1_1_outputs_conn_df = npf.get_and_norm_by_total_inputs(P1_1_outputs_conn_df,
                                                        c,
                                                        normalize_group_col='type_post',
                                                        groupby_cols=['type_pre', 'type_post', 'roi_noside', 
                                                        'instance_post', 'instance_pre']) #,

# inputs_to_P1_1b_outputs_neurons, inputs_to_P1_1b_outputs_conns = neu.fetch_adjacencies(
#                                           sources=None,
#                                           targets=NC(type=P1_1b_outputs_conn_df['type_post'].unique()),
#                                           client=c, min_total_weight=10)
# inputs_to_P1_1b_outputs_conns = neu.merge_neuron_properties(inputs_to_P1_1b_outputs_neurons, 
#                                                     inputs_to_P1_1b_outputs_conns, ['type', 'instance'])
# # Normalize P1_1b outputs by total outputs they get from ALL sources
# inputs_to_P1_1b_outputs_conns = normalize_weights_by_total(inputs_to_P1_1b_outputs_conns, 
#                                                            group_col='instance_post')
# 
# # Update output conn_df with normalized weights
# P1_1b_outputs_conn_df = inputs_to_P1_1b_outputs_conns[inputs_to_P1_1b_outputs_conns['type_pre']=='P1_1b'].copy() #groupby('type_post')['weight'].sum()
 

#%%
# Combine connection matrices
# ------------------------------------------------------------
weight_type = 'percent_of_total'
#weight_type = 'weight'
use_log_weights = weight_type == 'weight'
weight_label = f'log({weight_type}' if use_log_weights else weight_type.replace('_', ' ') 
vmax=0.1 if weight_type == 'percent_of_total' else None

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


P1_1_in, P1_1_out, P1_in_out = npf.matmul_conn_matrices(P1_1_inputs_conn_df, P1_1b_outputs_conn_df, 
                                 weight_label=weight_type,
                                 sort_rows='weight', sort_cols='weight',
                                 conn1_pre='type_pre', conn1_post='instance_post',
                                 conn2_pre='instance_pre', conn2_post='type_post',
                                 return_all=True)

if use_log_weights:
    P1_1_in = util.log_weights(P1_1_in)
    P1_1_out = util.log_weights(P1_1_out)
    P1_in_out = util.log_weights(P1_in_out)

#% PLOT
# Make a big grid of plots using GridSpec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2)
                      #width_ratios=[1, 1], height_ratios=[1, 1])
axn = [fig.add_subplot(gs[0, 0]), 
       fig.add_subplot(gs[0, 1]), 
       fig.add_subplot(gs[1:, 0:])] #, fig.add_subplot(gs[1, 1])]
npf.plot_connection_matrix(P1_1_in, ax=axn[0],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_type,
                       normalize_colors=True,
                       show_all_row_labels=True)
axn[0].set_title('P1_1b inputs (% of total inputs to P1_1b)')
npf.plot_connection_matrix(P1_1_out, ax=axn[1],
                       vmin=vmin, vmax=vmax,
                       colorbar_label=weight_type,
                       normalize_colors=True)
axn[1].set_title('P1_1b outputs (% total inputs to targets)')

axn[2].set_title('P1_1b inputs X outputs')
npf.plot_connection_matrix(P1_in_out, ax=axn[2],
                       vmin=vmin, vmax=None,
                       colorbar_label=weight_type,
                       normalize_colors=True,
                       show_all_row_labels=True,
                       show_all_col_labels=True)             
axn[2].set_title('P1_1b inputs X outputs')

putil.label_figure(fig, figid)
figname = f'P1_1_inputs_x_outputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%
# Which LC10a sources go to which P1, via P1_1?
# Level 2
P1_target_cols = [c for c in P1_in_out.columns if c.startswith('P1_')]
print(f"{len(P1_target_cols)}")

# get subset
LC10a_in_out = P1_in_out.loc['LC10a', P1_target_cols].copy()
LC10a_in_out.sort_values(ascending=False, inplace=True)

# plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=LC10a_in_out.index, y=LC10a_in_out.values, ax=ax,
                color=bg_color )
ax.set_title('LC10a inputs to P1 via P1_1')
# rotate x-tick labels
plt.xticks(rotation=90)
ax.set_ylabel(weight_label.replace('_', ' '))
plt.subplots_adjust(bottom=0.3, left=0.2)

putil.label_figure(fig, figid)
figname = f'LC10a_inputs_to_P1_via_P1_1_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%

# Get all P1 INPUTS:
# ------------------------------------------------------------
P1_inputs_neuron_df, P1_inputs_conn_df = neu.fetch_adjacencies(sources=None,
                                                               targets=NC(type='P1.*'),
                                                               min_total_weight=10)
P1_inputs_conn_df = neu.merge_neuron_properties(P1_inputs_neuron_df, P1_inputs_conn_df, ['type', 'instance'])
P1_inputs_conn_df = npf.extract_side_from_conn_df(P1_inputs_conn_df)

# Group across side
P1_inputs = P1_inputs_conn_df.groupby(['type_pre', 'type_post'], \
                                        as_index=False)['weight'].sum()

# Normalize by total inputs to each target type
P1_inputs = npf.norm_by_specified_inputs(P1_inputs, group_col='type_post')


#%%
# Get ALL P1 OUTPUTS:
# ------------------------------------------------------------
P1_outputs_neuron_df, P1_outputs_conn_df = neu.fetch_adjacencies(sources=NC(type='P1.*'), 
                                                                 targets=None)
P1_outputs_conn_df = neu.merge_neuron_properties(P1_outputs_neuron_df, P1_outputs_conn_df, ['type', 'instance'])
P1_outputs_conn_df = npf.extract_side_from_conn_df(P1_outputs_conn_df)
#%
# Group across side
P1_outputs = P1_outputs_conn_df.groupby(['type_pre', 'type_post'], \
                                        as_index=False)['weight'].sum()

#%%
# Normalize by total outputs to each target type
P1_outputs = npf.get_and_norm_by_total_inputs(P1_outputs, c,
                                          #groupby_type=True,
                                          normalize_group_col='type_post',
                                          )

#%% 
# P1 INPUTS:  Plot input matrix
clear_empty_cells = True
topN = 50
min_input_weight = 0.01
P1_input_conn_matrix = connection_table_to_matrix(P1_inputs,
                                    weight_col='percent_of_total',
                                    group_cols=['type_pre', 'type_post'],
                                    sort_by= ['type_pre', 'type_post'])

# Manually sort P1 labels (filter out None values)
pre_order = P1_input_conn_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
post_order = sorted(P1_inputs['type_post'].unique(), key=util.natsort)
P1_input_conn_matrix = P1_input_conn_matrix.reindex(index=pre_order, columns=post_order)

# Only take top N rows
P1_input_conn_filt = P1_input_conn_matrix.loc[pre_order[0:topN]]
# Plot
if clear_empty_cells:
    P1_input_conn_filt[P1_input_conn_filt==0] = np.nan
    
vmin = min_input_weight
vmax = 0.2
colorbar_label = 'percent of total inputs'
# Plot P1 input matrix
#fig, ax = plt.subplots(figsize=(6, 15))
fig = npf.plot_connection_matrix(P1_input_conn_filt, ax=None, #ax,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True, 
                       grid_color='k', grid_lw=0.01)
fig.axes[0].set_xlabel('Post-synaptic P1 type')
fig.axes[0].set_title('Top {} P1 inputs (min weight: {})'.format(topN, min_input_weight))

# save
putil.label_figure(fig, figid)
figname = f'P1_inputs_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%
# Plot all P1 outputs
P1_output_conn_matrix = connection_table_to_matrix(P1_outputs,
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
clear_empty_cells = True
topN = 60
min_output_weight = 0.05
vmax = None

# Only take top N outputs
P1_output_conn_matrix_filt = P1_output_conn_matrix[post_order[0:topN]]

if clear_empty_cells:
    P1_output_conn_matrix_filt[P1_output_conn_matrix_filt==0] = np.nan
    
# Plot P1 output matrix
fig = npf.plot_connection_matrix(P1_output_conn_matrix_filt, ax=None, #ax,
                       vmin=vmin, vmax=None,
                       colorbar_label=colorbar_label,
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=True,
                       grid_color='k', grid_lw=0.01)
fig.axes[0].set_xlabel('Post-synaptic P1 type')
fig.axes[0].set_title('Top {} P1 outputs (min weight: {})'.format(topN, min_output_weight))

# Highlight all outputs of P1_1b, where value greater than 0
P1_1b_outputs = P1_output_conn_matrix_filt.loc['P1_1b'].sort_values(ascending=False)
top_P1_1b_outputs = P1_1b_outputs.iloc[0:3]
npf.highlight_row_or_column(fig.axes[0], P1_output_conn_matrix_filt, 
                        column_label=top_P1_1b_outputs.index.tolist(), 
                        color='red', linewidth=1)



#%%

#%%
#%%

# Get P1 inputs matrix
weight_var = 'percent_of_total'
P1_inputs_matrix = P1_inputs.pivot(index='type_pre', columns='type_post', 
                                   values=weight_var).fillna(0)

assert P1_inputs_matrix.shape[0]==P1_inputs['type_pre'].unique().shape[0]
assert P1_inputs_matrix.shape[1]==P1_inputs['type_post'].unique().shape[0]
#%%
# P1 outputs matrix
P1_outputs_matrix = P1_outputs.pivot(index='type_pre', columns='type_post', 
                                   values=weight_var).fillna(0)

assert P1_outputs_matrix.shape[0]==P1_outputs['type_pre'].unique().shape[0]
assert P1_outputs_matrix.shape[1]==P1_outputs['type_post'].unique().shape[0]

#%% Combine

P1_combined = np.concatenate((P1_inputs_matrix.T, P1_outputs_matrix), axis=1)

P1_combined = pd.DataFrame(P1_combined,
                           index=P1_outputs_matrix.index.tolist(),
                           columns=P1_inputs_matrix.index.tolist() + P1_outputs_matrix.columns.tolist())

print(P1_combined.shape)

#%%
#mat_to_cluster = P1_outputs_matrix.copy()
cluster_inputs = False
output_type = 'in_out'
use_log_weights = False

label_cols = True
label_rows = False
if cluster_inputs:
    mat_to_cluster = P1_inputs_matrix.copy()
    rows = mat_to_cluster.index.tolist()
    cols = mat_to_cluster.columns.tolist()
    mat_to_cluster = mat_to_cluster.values
    plot_label = 'P1 inputs'
else:
    if output_type == 'in_out':
        mat_to_cluster = P1_combined.copy()
        plot_label = 'P1 in/outputs'
    else:
        mat_to_cluster = P1_outputs_matrix.copy()
        plot_label = 'P1 outputs'

    # Make sure the rows are what we are clustering (outputs)
    #mat_to_cluster = P1_combined.copy()    
    cols = mat_to_cluster.index.tolist()
    rows = mat_to_cluster.columns.tolist()
    mat_to_cluster = mat_to_cluster.T.values

 
cluster, dmat = hier_cosine( mat_to_cluster, distance_thresh=1)
z = linkage_order(cluster)

# Reorder the matrix based on clustering
print(len(z), mat_to_cluster.shape)
clustered_mat = mat_to_cluster[z, :] 

clustered_mat = pd.DataFrame(clustered_mat, index=rows, columns=cols)
print(clustered_mat.shape)

if use_log_weights:
    clustered_mat = util.log_weights(clustered_mat)
    vmin = clustered_mat.min().min()
    vmax = clustered_mat.max().max()
    colorbar_label = 'log(weight)'
else:
    vmin = 0.0
    vmax = 0.1
    colorbar_label = 'weight'
# Plot
fig, ax = plt.subplots(figsize=(6, 6))
npf.plot_connection_matrix(clustered_mat, ax=ax, vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       show_all_col_labels=label_cols,
                       show_all_row_labels=label_rows,
                       normalize_colors=True, show_grid=False)
fig.axes[0].set_title(f'{plot_label} (cosine similarity clustered)')

#%%
# Hierarchically cluster P1_in_mat using cosine similarity
P1_in_mat_clustered2, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(\
                                        P1_inputs_matrix, method='single')
if use_log_weights:
    P1_in_mat_clustered2 = util.log_weights(P1_in_mat_clustered2)
    vmin = P1_in_mat_clustered2.min().min()
    vmax = P1_in_mat_clustered2.max().max()
    colorbar_label = 'log(weight)'
else:
    vmin = 0.0
    vmax = 0.1
    colorbar_label = 'weight'
# plot
fig, ax = plt.subplots(figsize=(6, 6))
npf.plot_connection_matrix(P1_in_mat_clustered2, ax=ax, 
                       normalize_colors=True, show_grid=False,
                       vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       show_all_col_labels=True)
ax.set_title('P1 inputs (cosine similarity clustered)')
ax.set_xlabel('Post-synaptic P1 type')
ax.set_ylabel('Pre-synaptic P1 type')

# Optional: If you want to plot dendrograms, you can use scipy.cluster.hierarchy.dendrogram
# from scipy.cluster.hierarchy import dendrogram
# fig_dendro, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# dendrogram(cluster.linkage_, ax=ax1, labels=P1_inputs_matrix.index[z])
# ax1.set_title('Row Dendrogram')
# ax1.set_xlabel('P1 Input Types')







#%%

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

weight_type = 'percent_of_total'
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
npf.plot_connection_matrix(P1_P1_conn_matrix, ax=ax,
                       vmin=vmin, vmax=vmax,
                       colorbar_label='percent of total',
                       normalize_colors=True,
                       show_all_col_labels=True,
                       show_all_row_labels=True, show_grid=False, 
                       grid_color=[0.8]*3, grid_lw=0.001)
ax.set_aspect(1)
ax.set_title('P1_P1 connections')
ax.set_xlabel('Post-synaptic P1 type')
ax.set_ylabel('Pre-synaptic P1 type')

putil.label_figure(fig, figid)
figname = f'P1_P1_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%
# Cluster P1_P1_conn_matrix using cosine similarity
P1_P1_clustered, row_linkage, col_linkage = npf.cluster_matrix_cosine_similarity(P1_P1_conn_matrix, method='ward')

# Plot clustered matrix
fig = npf.plot_connection_matrix(P1_P1_clustered,
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

putil.label_figure(fig, figid)
figname = f'P1_P1_clustered_{weight_type}'
plt.savefig(os.path.join(output_dir, figname + '.png'), dpi=300)
print(figname)

#%%
# Plot dendrograms to understand the clustering
fig_dendro = npf.plot_dendrograms(row_linkage, col_linkage, 
                             row_labels=P1_P1_clustered.index.tolist(),
                             col_labels=P1_P1_clustered.columns.tolist())


#%%
# Test thresholded clustering (focuses on strong connections)
print("Testing thresholded clustering...")

# Method 1: Percentile-based thresholding (keep top 85% of connections per row/column)
P1_P1_clustered_thresh, row_linkage_thresh, col_linkage_thresh, matrix_thresh = \
    npf.cluster_matrix_cosine_similarity_thresholded(P1_P1_conn_matrix, 
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
    fig_analysis, row_clusters, col_clusters = npf.plot_cluster_analysis(
        P1_P1_clustered_thresh, row_linkage_thresh, col_linkage_thresh, 
        n_clusters=5, figsize=(20, 14), grid_lw=0, 
        show_all_labels=True, label_fontsize=8)
else:
    fig_analysis, row_clusters, col_clusters = npf.plot_cluster_analysis(
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