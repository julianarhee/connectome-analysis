'''
 # @ Author: Juliana Rhee
 # @ Filename:
 # @ Create Time: 2025-10-13 11:22:52
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-10-13 11:22:57
 # @ Description:
 '''

import os
import numpy as np
import pandas as pd


def get_upstream_synapses(root_id, connections):
    return connections[connections['post_root_id'] == root_id]
    
def get_upstream_neuropil_distribution(root_id):
    # get upstream synapses
    upstream_synapses = get_upstream_synapses(root_id)
    # get upstream neuropils
    neuropils = upstream_synapses['neuropil'].values
    synapse_counts = upstream_synapses['syn_count'].values
    # repeat neuropils according to synapse counts
    neuropils = np.repeat(neuropils, synapse_counts)
    return np.unique(neuropils, return_counts=True)

def get_primary_upstream_neuropil(root_id):
    neuropils, counts = get_upstream_neuropil_distribution(root_id)
    if len(neuropils) == 0:
        return ''
    return neuropils[np.argmax(counts)]

def get_side_from_name(name):
    if '_L' in name:
        return 'left'
    if '_R' in name:
        return 'right'
    return 'center'

#%

def get_neuron_class(root_id, classification):
    return classification[classification['root_id'] == root_id]['class'].values[0]

def simplify_cell_type(cell_type):
    # if the cell type has multiple ones separated by a comma, split them
    if ',' not in cell_type:
        return cell_type
    else:
        units = [x.strip() for x in cell_type.split(',')]
        # they might have a the initial part in common, find it
        common = os.path.commonprefix(units)
        # remove the common part from each unit and then join them using '/'
        return common + '/'.join([u.replace(common, '') for u in units])

def get_cell_type(root_id, classification):
    hemibrain_type = classification[classification['root_id'] == root_id]['hemibrain_type'].values[0]
    if hemibrain_type == '':
        cell_type = classification[classification['root_id'] == root_id]['cell_type'].values[0]
        return simplify_cell_type(cell_type)
    else:
        return simplify_cell_type(hemibrain_type)


# find connections from one set of roots to another
def find_connections(pre_root_ids, post_root_ids, connections):
    connections_from_pre = connections[connections['pre_root_id'].isin(pre_root_ids)].copy().reset_index(drop=True)
    connections_from_pre = connections_from_pre[connections_from_pre['post_root_id'].isin(post_root_ids)]
    # merge connections from same pre and post neurons
    connections_from_pre = connections_from_pre.groupby(['pre_root_id', 'post_root_id']).agg({'syn_count': 'sum'}).reset_index()
    return connections_from_pre

def find_connection_strengths(pre_root_ids, post_root_ids, connections):
    connections_from_pre = connections[connections['pre_root_id'].isin(pre_root_ids)].copy().reset_index(drop=True)
    connections_from_pre = connections_from_pre[connections_from_pre['post_root_id'].isin(post_root_ids)]
    # merge connections from same pre and post neurons
    connections_from_pre = connections_from_pre.groupby(['pre_root_id', 'post_root_id']).agg({'syn_strength': 'sum'}).reset_index()
    return connections_from_pre

def get_cell_side(root_id, classification):
    return classification[classification['root_id'] == root_id]['side'].values[0]

# find the roots by cell type
def find_roots_by_cell_type(cell_type, classification):
    classifieds = classification[classification['clean_cell_type'] == cell_type]
    return classifieds['root_id'].values

#%
def create_connection_matrix(pre_neurons, post_neurons, connections):
    ''' 
    Modified from Rishika's func.
    Create a connection matrix between pre and post neurons from a DataFrame of connections.
    '''
    # Filter the connections to only those that involve the specified pre and post neurons
    filtered_connections = connections[
        connections['pre_root_id'].isin(pre_neurons) & 
        connections['post_root_id'].isin(post_neurons)
    ]
    
    # Group by the neuron pair and sum the synapse counts
    grouped = filtered_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    
    # Pivot the grouped data to form the connection matrix
    connection_matrix_df = grouped.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    
    # Ensure the matrix rows/columns align with pre_neurons and post_neurons
    connection_matrix_df = connection_matrix_df.reindex(index=pre_neurons, columns=post_neurons, fill_value=0)
    
    # Convert to a numpy array
    connection_matrix = connection_matrix_df.values
    return connection_matrix

def create_second_order_connection_matrix(pre_neurons, post_neurons, connections):
    # Filter connections for valid synapse strengths
    valid_connections = connections.copy()
    valid_connections = valid_connections[~np.isnan(valid_connections['syn_strength'])]
    valid_connections = valid_connections[(valid_connections['syn_strength'] > 0) & (valid_connections['syn_strength'] < 1)]
    
    # Determine downstream neurons from pre_neurons and upstream neurons to post_neurons
    downstream_connections = valid_connections[valid_connections['pre_root_id'].isin(pre_neurons)]
    downstream_neurons = downstream_connections['post_root_id'].unique()
    
    upstream_connections = valid_connections[valid_connections['post_root_id'].isin(post_neurons)]
    upstream_neurons = upstream_connections['pre_root_id'].unique()
    
    # The common intermediate neurons are the intersection of downstream and upstream neurons
    common_neurons = np.intersect1d(downstream_neurons, upstream_neurons)
    
    # Build pre->common matrix using groupby and pivot:
    pre_common_connections = connections[
        (connections['pre_root_id'].isin(pre_neurons)) &
        (connections['post_root_id'].isin(common_neurons))
    ]
    pre_common_group = pre_common_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    pre_common_matrix_df = pre_common_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    # Ensure the matrix rows/columns align with pre_neurons and common_neurons
    pre_common_matrix_df = pre_common_matrix_df.reindex(index=pre_neurons, columns=common_neurons, fill_value=0)
    pre_common_matrix = pre_common_matrix_df.values

    # Build common->post matrix:
    common_post_connections = connections[
        (connections['pre_root_id'].isin(common_neurons)) &
        (connections['post_root_id'].isin(post_neurons))
    ]
    common_post_group = common_post_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
    common_post_matrix_df = common_post_group.pivot(index='pre_root_id', columns='post_root_id', values='syn_count').fillna(0)
    common_post_matrix_df = common_post_matrix_df.reindex(index=common_neurons, columns=post_neurons, fill_value=0)
    common_post_matrix = common_post_matrix_df.values

    # Multiply the matrices to obtain the second order connection matrix and take the square root (geometric mean)
    second_order_connection_matrix = np.dot(pre_common_matrix, common_post_matrix)
    second_order_connection_matrix = np.sqrt(second_order_connection_matrix)
    
    return second_order_connection_matrix, common_neurons


def flatten(xss):
    return [x for xs in xss for x in xs]

def get_neuron_name(root_id, names): 
    return names[names['root_id'] == root_id]['name'].values[0]


def load_visual_columns_data(data_folder):
    visual_columns = pd.read_csv(os.path.join(data_folder, 'raw', 'column_assignment.csv.gz'))
    visual_columns.head()

    print("{} cell types".format(visual_columns['type'].nunique()))
    print("{} cells".format(visual_columns['root_id'].nunique()))
    print("{} columns".format(visual_columns['column_id'].nunique()))

    return visual_columns


#%%

from collections import defaultdict
def parse_synapse_table(unzipped_file_path):
    pre_to_rows, post_to_rows = defaultdict(list), defaultdict(list)
    with open(unzipped_file_path, 'r') as file:
        next(file)  # Skip header row
        pre_id, post_id = None, None
        for line in file:
            parts = line.strip().split(',')
            pre_id = np.int64(parts[0]) if parts[0] else pre_id
            post_id = np.int64(parts[1]) if parts[1] else post_id
            x, y, z = map(int, parts[2:5])
            row = [pre_id, post_id, x, y, z]
            pre_to_rows[pre_id].append(row)
            post_to_rows[post_id].append(row)
    return pre_to_rows, post_to_rows

def load_synapse_coords_df(path_to_synapse_file, root_list, 
                      filename='synapse_coordinates_LC10a', 
                      save=True, create_new=False):
    '''
    Get synapse coordinates for a list of root_ids.
        
    Args:
    path_to_synapse_file: str, path to synapse coordinates file
    root_list: list, list of root_ids
    filename: str, name of file to save
    save: bool, save the dataframe
    create_new: bool, create a new dataframe
    ''' 
    processed_dir = os.path.split(path_to_synapse_file)[0].replace('raw', 'processed')   
    synapse_filepath = os.path.join(processed_dir, filename + '.csv')
    if not create_new and os.path.exists(synapse_filepath):
        syndf = pd.read_csv(synapse_filepath)
        print("Loaded: \n{}".format(synapse_filepath))
        return syndf

    print("Creating new synapse dataframe: {}".format(filename))    
    pre_to_rows, post_to_rows = parse_synapse_table(path_to_synapse_file)
    synapses_pre = pd.concat([pd.DataFrame(pre_to_rows[r], 
                                columns=['pre_root_id', 'post_root_id', 'x', 'y', 'z']) 
                            for r in root_list])
    synapses_pre['pre_post'] = 'pre'
    synapses_pre = synapses_pre.rename(columns={'pre_root_id': 'root_id'})
    
    synapses_post = pd.concat([pd.DataFrame(post_to_rows[r], 
                                columns=['pre_root_id', 'post_root_id', 'x', 'y', 'z'])
                            for r in root_list])
    synapses_post['pre_post'] = 'post'
    synapses_post = synapses_post.rename(columns={'post_root_id': 'root_id'})
    
    # combine into 1
    syndf = pd.concat([synapses_pre[['root_id', 'x', 'y', 'z', 'pre_post']], 
                       synapses_post[['root_id', 'x', 'y', 'z', 'pre_post']] ])                      
    #syndf.head()
   
    if save:
        syndf.to_csv(synapse_filepath)
        print("Saved: \n{}".format(synapse_filepath) )

    return syndf

def get_synapse_loc_each_cell(syndf):
    '''
    Get axo/den location for each cell in synapse coords df. 
    Calls get_den_ax_loc() from util.py on each cell, returning 1 loc each.
    
    Returns:
    lc10_locs: pd.DataFrame, with columns ['x', 'y', 'z', 'type', 'cell', 'n_terminals']
    (n_terminals should just be 0, 1 for axon/dendrite, but in case there is >1 of either)
    '''
    # For each ID, get inputs (dendrite labels) and outputs (axon labels)
    d_list = []
    for cell, curr_syndf in syndf.groupby('root_id'):
        ax_term, den_term = get_den_ax_loc(int(cell), curr_syndf, num_gauss=2)
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
    lc10_locs = pd.concat(d_list, axis=0)
    lc10_locs = lc10_locs.reset_index().rename(columns={'index': 'n_terminals'})
    
    return lc10_locs

def get_den_ax_loc(cell, syndf, num_gauss=3):
    """
    (modeled after Charlie Dowell), adjusted for FlyWire synapse_coordinates.csv. 
    Tested with LC10a neurons - axon is "post" and dendrites are "pre".

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
    
    #syndf = neu.fetch_synapses(cell)
    pre_post = syndf[syndf['root_id']==cell]['pre_post']
    pre = pd.Series.to_numpy(pre_post=='pre')
    post = pd.Series.to_numpy(pre_post=='post')
    syn_locs = np.array([syndf['x'], syndf['y'], syndf['z']])
    syn_locs = np.transpose(syn_locs)
  
    if 'confidence' in syndf.columns:
        cdx = pd.Series.to_numpy(syndf['confidence']>0.9) # Can relax this if needs be
        pre_locs = syn_locs[pre&cdx,:]
        post_locs = syn_locs[post&cdx,:]
    else:
        pre_locs = syn_locs[pre,:]
        post_locs = syn_locs[post,:]
         
    # get mean pre location
    mn_pre = np.mean(pre_locs,axis=0)
    sd_pre = np.std(pre_locs,axis=0)
    #
    gm = mixture.GaussianMixture(n_components=num_gauss, random_state=0).fit(post_locs)
    # Gaussian mixture model with two gaussians
    mn_post = gm.means_
    
    ax_term = mn_pre
    den_dx = np.sqrt(np.sum(np.square(mn_post-mn_pre),axis=1))
    dx = [True]*num_gauss #[True, True, True]
    ax_dx = np.argmin(den_dx)
    dx[ax_dx] = False
    den_term = mn_post[dx,:]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(post_locs[:,0],post_locs[:,1],post_locs[:,2],color='k')
    # ax.scatter(pre_locs[:,0],pre_locs[:,1],pre_locs[:,2],color='r')
    # ax.scatter(mn_post[:,0],mn_post[:,1],mn_post[:,2],color='b')
    # plt.show()
    
    return ax_term, den_term

