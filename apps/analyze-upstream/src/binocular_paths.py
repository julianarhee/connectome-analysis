'''
 # @ Author: Juliana Rhee
 # @ Filename: binocular_paths.py
 # @ Create Time: 2025-10-09 11:43:24
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-10-09 11:43:36
 # @ Description: Analyze binocular paths to LC10a  
 '''

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotting as putil
import utils as util

#%%
# Set output dir
rootdir = '/Volumes/Juliana/connectome'
figdir = os.path.join(rootdir, 'analyses', 'binocular_paths')
if not os.path.exists(figdir):
    os.makedirs(figdir)
#

#%%
# data
data_folder = '../../../data/flywire'
flywire_datafiles = glob.glob(os.path.join(data_folder, '*.csv.gz'))
print(flywire_datafiles)
#os.listdir(data_folder)

figid = data_folder

#%%
# Load data
for f in flywire_datafiles:
    parentdir, fname = os.path.split(f)
    file_name = fname.split('.')[0]
    command = file_name+"= pd.read_csv('"+ os.path.join(data_folder, 'raw', f) +"')"
    exec(command)
    print(command)
#%%

def load_data_to_workspace():
    #%
    skip_connections = True

    for f in flywire_datafiles:
        parentdir, fname = os.path.split(f)
        file_name = fname.split('.')[0]
        print(file_name)
        if 'classification' in file_name:
            # root_id	flow	super_class	class	sub_class	hemilineage	side	nerve
            classification = pd.read_csv(f)
            continue
        if skip_connections and 'connections' in file_name:
            continue
        if 'synapse_coordinates' in file_name:
            continue
        if file_name == 'fafb_v783_princeton_synapse_table':
            continue
            #file_name = 'synapses'
        if file_name == 'consolidated_cell_types':
            # root_id	primary_type	additional_type(s)
            cell_types = pd.read_csv(f)
            continue
        if file_name == 'visual_neuron_types':
            # root_id	type	family	subsystem	category	side
            visual_neuron_types = pd.read_csv(f)
        command = file_name+"= pd.read_csv('"+ f +"')"
        exec(command)
        print(command)

    # target connections data
    # pre_root_id	post_root_id	neuropil	syn_count	nt_type
    target_connections = 'connections_princeton_no_threshold'
    connections = pd.read_csv(os.path.join(data_folder, target_connections + '.csv.gz'))
    
    return classification, connections, cell_types, visual_neuron_types

def load_synapses():
    #%
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
        
    return synapses

# %%
classification, connections, cell_types, visual_neuron_types = load_data_to_workspace()

# %%

#########print(visual_neuron_types['family'].unique())
min_syn_count = 5 
conns = connections[connections['syn_count']>=min_syn_count].copy()


### How many inputs to LC10a are left and right?
LC10a_ids = visual_neuron_types[visual_neuron_types['type']=='LC10a']['root_id'].unique()

lc10a_input_ids = conns[conns['post_root_id'].isin(LC10a_ids)]['pre_root_id'].unique()
binoc_lc10a_inputs = visual_neuron_types[(visual_neuron_types['root_id'].isin(lc10a_input_ids))
                    & (visual_neuron_types['side']=='left,right')].copy()
binoc_lc10a_input_ids = binoc_lc10a_inputs['root_id'].unique()
print("Binocular LC10a inputs:")
print(binoc_lc10a_inputs['type'].unique())


#%%
upstream_neurons = ['LC14a1']
LC14a1_ids = visual_neuron_types[visual_neuron_types['type'].isin(upstream_neurons)]['root_id'].unique()

upstream_ids = binoc_lc10a_input_ids

lc14_to_lc10a_conn = conns[conns['pre_root_id'].isin(LC14a1_ids) 
             & conns['post_root_id'].isin(LC10a_ids) ].copy()

n_lc14_targets = lc14_to_lc10a_conn['post_root_id'].nunique()
print(f"{n_lc14_targets} of {len(LC10a_ids)} LC10a neurons get LC14a1 inputs")

binoc_to_lc10a_conn = conns[conns['pre_root_id'].isin(binoc_lc10a_input_ids) 
             & conns['post_root_id'].isin(LC10a_ids) ].copy()
print(f"{binoc_to_lc10a_conn['post_root_id'].nunique()} of {len(LC10a_ids)} LC10a neurons get binocular inputs")

# %%

aotu_id_to_name = {720575940616012061: 'AOTU025_R',
                    720575940631517251: 'AOTU019_R',
                    720575940633556644: 'AOTU019_L',
                    720575940639182424: 'AOTU025_L'}

aotu019_ids = cell_types[cell_types['primary_type']=='AOTU019']['root_id'].unique()
aotu025_ids = cell_types[cell_types['primary_type']=='AOTU025']['root_id'].unique()
aotu_ids = list(aotu_id_to_name.keys())

# classification[classification['root_id'].isin(aotu019_ids)]

#%%
# Build connection matrix from LC14a1 -> LC10a -> AOTU019, split by L/R
import network as nwk

#%%
# Get first order connections, LC10a -> AOTU019/25
pre_neurons = LC10a_ids
post_neurons = aotu_ids
syn_var = 'syn_count'
matrix_file = 'test' # None
recalculate = True

# Group LC10a neurons by L/R
LC10a_L_ids = visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
                                  & (visual_neuron_types['side']=='left')]['root_id'].unique()
LC10a_R_ids = visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
                                  & (visual_neuron_types['side']=='right')]['root_id'].unique()

pre_neurons = np.concatenate([LC10a_L_ids, LC10a_R_ids])
pre_neuron_side = ['left']*len(LC10a_L_ids) + ['right']*len(LC10a_R_ids)
assert len(LC10a_L_ids)+len(LC10a_R_ids)==len(LC10a_ids)
#print(len(pre_neurons))
#%
# Get first order connections
LC10a_AOTU_matrix_first, LC10a_AOTU_paths_first = nwk.get_first_order(
                                                pre_neurons, 
                                                post_neurons, conns, 
                                                syn_var = syn_var,
                                                matrix_file=matrix_file, 
                                                recalculate=recalculate)
                                                #return_intermediates=True)  

# Plot
fig, ax = plt.subplots(figsize=(4, 10))
sns.heatmap(LC10a_AOTU_matrix_first, ax=ax, cmap='magma')
# Label axes with neuron names
ax.set_xticklabels(aotu_id_to_name.values(), fontsize=6)
ax.set_ylabel('LC10a neurons')
ax.set_yticks(np.arange(0.5, len(pre_neurons), 1))
#ax.set_yticklabels(pre_neurons, fontsize=6, rotation=0)
ax.set_yticklabels([v if i%5==0 else '' for i, v in enumerate(pre_neuron_side)], fontsize=6, rotation=0)
#ax.set_aspec
# Label colorbar
cbar = ax.collections[0].colorbar
cbar.set_label(syn_var, fontsize=12)
cbar.ax.tick_params(labelsize=12)
# Title
ax.set_title('LC10a -> AOTU019/25')

plt.show()

#%%


def split_neurons_by_side(pre_neurons, classification):
    # Split LC14a1 neurons by L/R
    #if pre_neuron_type == 'LC14a1':
    pre_neuron_L_ids = []
    pre_neuron_R_ids = []
    for i in pre_neurons:
        if classification[(classification['root_id']==i)]['side'].unique() == 'left':
            pre_neuron_L_ids.append(i)
        else:
            pre_neuron_R_ids.append(i)

    # elif pre_neuron_type == 'LC10a':
    #     
    #     # Group LC10a neurons by L/R
    #     pre_neuron_L_ids= visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
    #                                     & (visual_neuron_types['side']=='left')]['root_id'].unique()
    #     pre_neuron_R_ids = visual_neuron_types[(visual_neuron_types['type']=='LC10a') 
    #                                     & (visual_neuron_types['side']=='right')]['root_id'].unique()
    # Check 
    pre_neurons = np.concatenate([pre_neuron_L_ids, pre_neuron_R_ids])
    pre_neuron_side = ['left']*len(pre_neuron_L_ids) + ['right']*len(pre_neuron_R_ids)
    assert len(pre_neuron_L_ids)+len(pre_neuron_R_ids)==len(pre_neurons)


    return pre_neurons, pre_neuron_side


#%%
# LC14a1 -> LC10a -> AOTU019/25
pre_neuron_type = 'LC14a1'
post_neuron_type = 'LC10a'
connection_order = 1


# Assign pre/post
if pre_neuron_type == 'LC14a1':
    pre_neurons = LC14a1_ids
    ylabel = 'Bilateral LC14a1 neurons'
elif pre_neuron_type == 'LC10a':
    pre_neurons = LC10a_ids
    ylabel = 'LC10a neurons'
else:
    raise ValueError(f"Invalid pre_neuron_type: {pre_neuron_type}")

if post_neuron_type == 'AOTU':
    post_neurons = aotu_ids
    xlabel = 'AOTU0 neurons'
elif post_neuron_type == 'LC10a':
    post_neurons = LC10a_ids
    xlabel = 'LC10a neurons'
else:
    raise ValueError(f"Invalid pre_neuron_type: {pre_neuron_type}")

# Split by left/right
pre_neurons, pre_neuron_side = split_neurons_by_side(pre_neurons, classification)
post_neurons, post_neuron_side = split_neurons_by_side(post_neurons, classification)

if post_neuron_type == 'AOTU':
    xtick_labels = aotu_id_to_name.values()
    xlabel = 'AOTU0 neurons'
elif post_neuron_type == 'LC10a':
    xtick_labels = post_neuron_side
    xlabel = 'LC10a neurons'
else:
    raise ValueError(f"Invalid pre_neuron_type: {pre_neuron_type}")

recalculate = True
syn_var = 'syn_count'
matrix_file = 'test' # None

if connection_order == 2:
    # Get second order connections
    matrix_, paths_, _ = nwk.get_second_order(
                                                    pre_neurons, 
                                                    post_neurons, conns, 
                                                    syn_var = syn_var,
                                                    matrix_file=matrix_file, 
                                                    recalculate=recalculate,
                                                    return_intermediates=True)  
elif connection_order == 1:
    # Get first order connections
    matrix_, paths_ = nwk.get_first_order(
                                                    pre_neurons, 
                                                    post_neurons, conns, 
                                                    syn_var = syn_var,
                                                    matrix_file=matrix_file, 
                                                    recalculate=recalculate)
                                                    #return_intermediates=True)  
   
# Plot
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(matrix_, ax=ax)
# Label axes with neuron names
ax.set_xticks(np.arange(0.5, len(xtick_labels), 1))
if len(xtick_labels) > 10:
    ax.set_xticklabels([v if i%5==0 else '' for i, v in enumerate(xtick_labels)], fontsize=6)
else:
    ax.set_xticklabels(xtick_labels, fontsize=6)
ax.set_ylabel(ylabel)
ax.set_yticks(np.arange(0.5, len(pre_neurons), 1))
ax.set_yticklabels([v if i%5==0 else '' for i, v in enumerate(pre_neuron_side)], fontsize=6, rotation=0)
# Label colorbar
cbar = ax.collections[0].colorbar
cbar.set_label(syn_var, fontsize=12)
cbar.ax.tick_params(labelsize=12)
# Title
ax.set_title(f"{pre_neuron_type} -> {post_neuron_type}") #'LC14a1 -> LC10a -> AOTU019/25')

plt.show()

#%%


#%%
nwk.get_explicit_paths_sparse_second

conns[(conns['pre_root_id'].isin(lc14_to_lc10a_conn['post_root_id']))
      & (conns['post_root_id'].isin(aotu019_ids)) 
      ]

