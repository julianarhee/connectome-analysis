#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : lc10a_flywire.py
Created        : 2025/09/21 22:41:56
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/TuTu-LC1a0-AOTU/src
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

# Set matplotlib for inline plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# data
data_folder = '../../../flywire-data'
flywire_datafiles = glob.glob(os.path.join(data_folder, '*.csv.gz'))
print(flywire_datafiles)
#os.listdir(data_folder

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
synapses_outpath = '../../../flywire-data/fafb_synapses.parquet'
if os.path.exists(synapses_outpath):
    synapses = pd.read_parquet(synapses_outpath)
    print("Loaded synapses from pickle")
else:
    print("Creating synapses pickle")
    synapse_fpath = '../../../flywire-data/fafb_v783_princeton_synapse_table.csv.gz'
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
# 720575940640425294 TuTuAb (TuTuA_1?)
# 720575940622538520 TuTuAa (TuTuA_2?)

# 720575940628586261  TuTuTAa_L
# 720575940612218547 TuTuAb_L
# 720575940622538520 TuTuAa_R
# 720575940640425294  TuTuAb_R

tutu_names= {
            720575940628586261: 'TuTuTAa_L',
            720575940612218547: 'TuTuAb_L',
            720575940622538520: 'TuTuAa_R',
            720575940640425294: 'TuTuAb_R'}

tutu_right = [720575940622538520, 20575940640425294]
tutu_left = [720575940628586261, 720575940612218547]
tutu_types = ['TuTuAa', 'TuTuAb']
tutu_all = consolidated_cell_types[consolidated_cell_types['primary_type'].isin(tutu_types)]

#%%
#lc10a_all = consolidated_cell_types[consolidated_cell_types['primary_type']=='LC10a']
#len(lc10a_all)

# %%
side = 'right'
lc10a_neurons = visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
                  & (visual_neuron_types['side']==side)]
print(len(lc10a_neurons))

all_lc10a_neurons = visual_neuron_types[visual_neuron_types['type']=='LC10a']

#%% aoi
aotu_types = ['AOTU019', 'AOTU025']
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
# tutu to LC10a
min_syn_count = 5
tutu_lc10a_conn = find_connections(tutu_all['root_id'].unique(), 
                                   lc10a_neurons['root_id'].unique())
tutu_lc10a_conn_filt = tutu_lc10a_conn[tutu_lc10a_conn['syn_count']>min_syn_count]
tutu_lc10a_conn_filt.loc[tutu_lc10a_conn_filt['pre_root_id'].isin(tutu_right), 'side'] = 'right'
tutu_lc10a_conn_filt.loc[tutu_lc10a_conn_filt['pre_root_id'].isin(tutu_left), 'side'] = 'left'

print(len(tutu_lc10a_conn_filt))

# LC10a to AOTU
lc10a_aotu_conn = find_connections(lc10a_neurons['root_id'].unique(), 
                                    aotu_neurons['root_id'].unique())
lc10a_aotu_conn_filt = lc10a_aotu_conn[lc10a_aotu_conn['syn_count']>min_syn_count]
print(len(lc10a_aotu_conn_filt))

#%%

plot_all_lc10a = True

if plot_all_lc10a:
    lc10a_post = synapses[synapses['post_root_id'].isin(all_lc10a_neurons['root_id'].unique())]
else:
    lc10a_post = synapses[synapses['post_root_id'].isin(lc10a_neurons['root_id'].unique())]

hue_sortby = 'pre_y'
hue_ascending = True
# Sort LC10a_post by post_y
lc10a_post_sorted_ids = lc10a_post.sort_values(
                            by=hue_sortby, 
                            ascending=hue_ascending)['post_root_id'].unique()

# Create dictionary of colors
lc10a_colors = sns.color_palette('viridis', n_colors=len(lc10a_post_sorted_ids)).as_hex()
lc10a_cdict = dict(zip(lc10a_post_sorted_ids, lc10a_colors))

#%print(len(lc10a_post))
# Check current backend and data
print(f"Current matplotlib backend: {matplotlib.get_backend()}")
print(f"Data shape: {lc10a_post.shape}")
print(f"Number of unique root IDs: {lc10a_post['post_root_id'].nunique()}")


# Plot interactive 3D scatter plot
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot each LC10a neuron with its assigned color
print("Plotting neurons...")
for i, root_id in enumerate(lc10a_post['post_root_id'].unique()[0::3]):
    data_subset = lc10a_post[lc10a_post['post_root_id'] == root_id]
    color = lc10a_cdict[root_id]
    ax.scatter(data_subset['post_x'], data_subset['post_y'], data_subset['post_z'],
               c=color, s=20, alpha=0.7, label=root_id)
    if i % 10 == 0:  # Print progress every 10 neurons
        print(f"  Plotted {i+1}/{lc10a_post['post_root_id'].nunique()} neurons")

# Set labels and title
ax.set_xlabel('post_x')
ax.set_ylabel('post_y')
ax.set_zlabel('post_z')
ax.set_title('LC10a post-synaptic partners (Interactive 3D)')

# Enable interactive controls
ax.view_init(elev=20, azim=45)  # Set initial viewing angle

# Add legend (optional - might be cluttered with many neurons)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

print("Showing plot...")
# Show the interactive plot
plt.show()
print("3D plot displayed inline.")
#%%

# Plot 3D plot
# 3D plot
#fig = plt.figure(figsize=(10, 10))
fig, axn = plt.subplots(3, 1, figsize=(5, 8))
# Plot all combinations of post_x, post_y, post_z
for ai, (xvar, yvar) in enumerate([['post_x', 'post_y'], ['post_x', 'post_z'], ['post_y', 'post_z']]):
    ax = axn[ai]
    sns.scatterplot(data=lc10a_post, ax=ax,
                    x=xvar, y=yvar,# z='post_z', #z='z_pre', 
                    s=10,
                    hue='post_root_id', 
                    palette=lc10a_cdict, legend=0)
    ax.set_aspect('equal')
    ax.invert_yaxis()
plt.subplots_adjust(hspace=0.8, wspace=0.8)

fig.suptitle(f'Hue: {hue_sortby}, ascending: {hue_ascending}')

plt.show()

#%%a
# Just plot 1 hemisphere
lc10a_post = synapses[synapses['post_root_id'].isin(lc10a_neurons['root_id'].unique())]

hue_sortby = 'pre_y' #'pre_y'
if hue_sortby == 'pre_z':
    hue_ascending = False #    hue_ascending = True
    hue_palette = 'viridis'

    xvar = 'post_y'
    yvar = 'post_z'
    # can also be 
    # xvar='post_x'
    # yvar = 'post_y
elif hue_sortby == 'pre_y':
    hue_ascending = True
    hue_palette = 'viridis_r'

    xvar = 'post_z'
    yvar = 'post_y'

# Sort LC10a_post by post_y
lc10a_post_sorted_ids = lc10a_post.sort_values(
                            by=hue_sortby, 
                            ascending=hue_ascending)['post_root_id'].unique()

# Create dictionary of colors
lc10a_colors = sns.color_palette(hue_palette, n_colors=len(lc10a_post_sorted_ids)).as_hex()
lc10a_cdict = dict(zip(lc10a_post_sorted_ids, lc10a_colors))


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
if plot_all_lc10a:
    ax.set_title("LC10a post-synaptic partners (hue sort_by={})".format(hue_sortby), loc='left')
else:
    ax.set_title("LC10a ({}) post-synaptic partners".format(side), loc='left')
sns.scatterplot(data=lc10a_post, ax=ax,
                x=xvar, y=yvar, #z='z_pre', 
                hue='post_root_id', 
                palette=lc10a_cdict, legend=0)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.invert_xaxis()

#%%
aotu_ids = lc10a_aotu_conn['post_root_id'].unique() #aotu_neurons['root_id'].unique()
aotu_colors = sns.color_palette('viridis', n_colors=len(aotu_ids)+1).as_hex()
aotu_cdict = {720575940616012061: aotu_colors[-1],
              720575940631517251: aotu_colors[0]}

aotu_names = {'AOTU025': 720575940616012061,
              'AOTU019': 720575940631517251}
aotu_name_cdict = {'AOTU019': aotu_cdict[aotu_names['AOTU019']],
                   'AOTU025': aotu_cdict[aotu_names['AOTU025']]}

aotu_name_types = {v: k for k, v in aotu_names.items()}
aotu_name_types

# %%
aotu_ids = aotu_neurons['root_id'].unique()
aotu_post = synapses[(synapses['post_root_id'].isin(aotu_ids)) 
                   & (synapses['pre_root_id'].isin(lc10a_neurons['root_id'].unique()))]
print(len(aotu_post))
#%%
aotu_post.loc[aotu_post['post_root_id'] == aotu_names['AOTU019'], 'aotu_type'] = 'AOTU019'
aotu_post.loc[aotu_post['post_root_id'] == aotu_names['AOTU025'], 'aotu_type'] = 'AOTU025'


#%%
#xvar = 'post_x'
#yvar = 'post_z'
fig, axn = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
ax=axn[0]
ax.set_title("LC10a->AOTU post-synaptic partners", loc='left')
sns.scatterplot(data=aotu_post, ax=ax,
                x=xvar, y=yvar, #z='z_pre', 
                hue='pre_root_id', s=10,
                palette=lc10a_cdict, legend=0)
ax.set_aspect('equal')
ax=axn[1]
sns.scatterplot(data=aotu_post, ax=ax,
                x=xvar, y=yvar, #z='z_pre', 
                hue='aotu_type', s=10,
                palette=aotu_name_cdict, legend=1)
ax.set_aspect('equal')
sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1.1), frameon=False)
ax.invert_yaxis()
plt.show()


# %%
tutu_lc10a_syn = synapses[synapses['pre_root_id'].isin(tutu_all['root_id'].unique()) & 
                          synapses['post_root_id'].isin(lc10a_neurons['root_id'].unique())]
print(len(tutu_lc10a_syn))

lc10a_aotu_syn = synapses[synapses['pre_root_id'].isin(lc10a_neurons['root_id'].unique()) & 
                          synapses['post_root_id'].isin(aotu_neurons['root_id'].unique())]
print(len(lc10a_aotu_syn))

# %%
tutu_ids = tutu_lc10a_syn['pre_root_id'].unique()
tutu_colors = sns.color_palette('colorblind', n_colors=len(tutu_ids)).as_hex()
tutu_cdict = dict(zip(tutu_ids, tutu_colors))

#xvar = 'post_x'
#yvar = 'post_y'
# Plot TuTu
fig, axn = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
ax=axn[0]
ax.set_title("TuTu -> LC10a", loc='left')
sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                x=xvar, y=yvar, #z='z_pre', 
                hue='pre_root_id', 
                palette=tutu_cdict)
ax.set_aspect('equal')
ax=axn[1]
ax.set_title('Colored by LC10a position', loc='left')
sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                x=xvar, y=yvar, #z='z_pre', 
                hue='post_root_id', 
                palette=lc10a_cdict, legend=0)
ax.set_aspect('equal')
ax.invert_yaxis()
plt.show()

#%%
#xvar = 'pre_x'
#yvar = 'pre_y'
xvar_pre = xvar.replace('post_', 'pre_')
yvar_pre = yvar.replace('post_', 'pre_')

for k, v in tutu_names.items():
    tutu_lc10a_syn.loc[tutu_lc10a_syn['pre_root_id'] == k, 'tutu_name'] = v

# Plot all TuTu subtypes
n_tutu_types = len(tutu_all['root_id'].unique())
fig, axn = plt.subplots(1, n_tutu_types, figsize=(10, 3), sharex=True, sharey=True)

for ti, (tutu_type, tutu_) in enumerate(tutu_lc10a_syn.groupby('tutu_name')): #'pre_root_id')):
    print(ti, len(tutu_))
    ax=axn[ti]
    sns.scatterplot(data=tutu_, ax=ax,
                    x=xvar_pre, y=yvar_pre, #z='z_pre', 
                    hue='pre_root_id', 
                    palette=tutu_cdict, legend=0)
    ax.set_aspect('equal')
    tutu_name = ''
    ax.set_title(tutu_type, fontsize=6, loc='left')
ax.invert_yaxis()

fig.suptitle('TuTu -> LC10a', fontsize=10)


# %%
# Find TuTu->LC10a->AOTU19 and TuTu->LC10a->AOTU25
aotu19_id = 720575940631517251
aotu25_id = 720575940616012061

lc10a_aotu19_ids = lc10a_aotu_conn_filt[lc10a_aotu_conn_filt['post_root_id'] == aotu19_id]['pre_root_id'].unique()
lc10a_aotu25_ids = lc10a_aotu_conn_filt[lc10a_aotu_conn_filt['post_root_id'] == aotu25_id]['pre_root_id'].unique()

#%%
tutu_lc10a_syn.loc[tutu_lc10a_syn['post_root_id'].isin(lc10a_aotu19_ids), 'aotu_type'] = 'AOTU19'
tutu_lc10a_syn.loc[tutu_lc10a_syn['post_root_id'].isin(lc10a_aotu25_ids), 'aotu_type'] = 'AOTU25'

#%%
aotu_colors = {'AOTU19': aotu_cdict[aotu19_id], 'AOTU25': aotu_cdict[aotu25_id]}
#%
# Plot TuTu->LC10a->AOTU19
fig, axn = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
ax=axn[0]
ax.set_title("TuTu->LC10a->AOTU19 (colored by LC10a pos)", loc='left')
sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                x=xvar_pre, y=yvar_pre, #z='z_pre', 
                hue='post_root_id', 
                palette=lc10a_cdict, legend=0)
ax.set_aspect('equal')
ax=axn[1]
ax.set_title("", loc='left')
sns.scatterplot(data=tutu_lc10a_syn, ax=ax,
                x=xvar_pre, y=yvar_pre, #z='z_pre', 
                hue='aotu_type', alpha=0.5, 
                palette=aotu_colors)
ax.set_aspect('equal')

ax.invert_yaxis()
# %%
# Count synapses from TuTu to LC10a to AOTU19 and AOTU25

lc10a_aotu19_ids = lc10a_aotu_conn_filt[lc10a_aotu_conn_filt['post_root_id'] == aotu19_id]['pre_root_id'].unique()
lc10a_aotu25_ids = lc10a_aotu_conn_filt[lc10a_aotu_conn_filt['post_root_id'] == aotu25_id]['pre_root_id'].unique()
# Find common
common_lc10a_neurons = list(set(lc10a_aotu19_ids) & set(lc10a_aotu25_ids))
print(f"Number of common LC10a neurons: {len(common_lc10a_neurons)}")

#%
#tutu_lc10a_syn.loc[tutu_lc10a_syn['post_root_id'].isin(lc10a_aotu19_ids), 'aotu_type'] = 'AOTU19'
#tutu_lc10a_syn.loc[tutu_lc10a_syn['post_root_id'].isin(lc10a_aotu25_ids), 'aotu_type'] = 'AOTU25'
tutu_lc10a_syn['syn_count_aggr'] = 0

tutu_list = []
for aotu_id in [aotu19_id, aotu25_id]:
    
    lc10a_to_aotu = lc10a_aotu_conn_filt[lc10a_aotu_conn_filt['post_root_id'] == aotu_id]['pre_root_id'].unique()
    print(f"Number of LC10a neurons that go to {aotu_id}: {len(lc10a_to_aotu)}")

    tutu_lc10a_syn_curr = tutu_lc10a_syn[tutu_lc10a_syn['post_root_id'].isin(lc10a_to_aotu)]
    #print(f"Number of TuTu neurons that go to {aotu_id}: {len(tutu_lc10a_syn_curr)}")

    for curr_lc10a_id, tutu_ in tutu_lc10a_syn_curr.groupby('post_root_id'):
        
        # Get N synapses from TuTu to this LC10a neuron
        tutu_counts = tutu_lc10a_conn_filt[tutu_lc10a_conn_filt['post_root_id']==curr_lc10a_id] #= 'AOTU19'

        print(curr_lc10a_id, len(tutu_counts))

        # Get N synapses from this  LC10a to this AOTU
        lc10a_aotu_counts = lc10a_aotu_conn_filt[(lc10a_aotu_conn_filt['pre_root_id'] == curr_lc10a_id)
                                            & (lc10a_aotu_conn_filt['post_root_id'] == aotu_id)]['syn_count'].unique()[0] #.sum()
        print(curr_lc10a_id, lc10a_aotu_counts) 
        tutu_['count_post_lc10a'] = lc10a_aotu_counts
        tutu_['aotu_type'] = aotu_name_types[aotu_id]
        total_counts = lc10a_aotu_counts + tutu_counts['syn_count'].sum()
        tutu_['count_post_aggr'] = total_counts
        tutu_list.append(tutu_)
        tutu_lc10a_syn.loc[
                          (tutu_lc10a_syn['post_root_id'] == curr_lc10a_id), 'syn_count_aggr'] = total_counts

tutu_lc10a_aotu_syn_aggr = pd.concat(tutu_list, ignore_index=True)



#%%
# Plot TuTu->LC10a->AOTU19 and TuTu->LC10a->AOTU25
# Hue and size by count_post_aggr
xvar_post = 'post_z'
yvar_post = 'post_y'

# Normalize size of synapse counts
size_min = tutu_lc10a_aotu_syn_aggr['syn_count_aggr'].min()
size_max = tutu_lc10a_aotu_syn_aggr['syn_count_aggr'].max()
print(f"syn_count_aggr range: {size_min} to {size_max}")
print(f"Unique syn_count_aggr values: {sorted(tutu_lc10a_syn['syn_count_aggr'].unique())}")


# Create a function to normalize sizes consistently across all plots
def normalize_size(value, min_val, max_val, size_range=(20, 300)):
    """Normalize a value to size range based on global min/max"""
    if max_val == min_val:
        return size_range[0]
    normalized = (value - min_val) / (max_val - min_val)
    return size_range[0] + normalized * (size_range[1] - size_range[0])

# Add normalized size columns to main dataframe
size_range = (1, 100)
tutu_lc10a_aotu_syn_aggr['norm_size'] = tutu_lc10a_aotu_syn_aggr['syn_count_aggr'].apply(
    lambda x: normalize_size(x, size_min, size_max, size_range)
)

# Normalize size of synapse counts
fig, axn = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
ax=axn[0]
ax.set_title("TuTu->LC10a->AOTU19 (colored by LC10a pos)", loc='left')
sns.scatterplot(data=tutu_lc10a_aotu_syn_aggr, ax=ax,
                x=xvar_post, y=yvar_post, #z='z_pre', 
                hue='post_root_id', size='norm_size', sizes=size_range,
                palette=lc10a_cdict, legend=0)
ax.set_aspect('equal')
ax=axn[1]
sns.scatterplot(data=tutu_lc10a_aotu_syn_aggr, ax=ax,
                x=xvar_post, y=yvar_post, #z='z_pre', 
                hue='aotu_type', #size='norm_size', sizes=size_range,
                alpha=0.3, 
                palette=aotu_colors, legend=1)
ax.set_aspect('equal')

ax=axn[2]
ax.set_title("", loc='left')
sns.scatterplot(data=tutu_lc10a_aotu_syn_aggr, ax=ax,
                x=xvar_post, y=yvar_post, #z='z_pre', 
                hue='syn_count_aggr', size='norm_size', sizes=size_range,
                alpha=0.5, 
                palette='viridis')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.1), frameon=False)
ax.set_aspect('equal')
ax.invert_yaxis()
plt.show()

# %%