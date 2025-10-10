#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Mar 9 11:06:00 2024
@File    :   utils.py
@Time    :   2024/05/09 10:07:48
@Author  :   julianarhee, rishikamohanta
'''
import os
import numpy as np
import pandas as pd


from neuprint import Client
def set_default_clien():
    c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imp1bGlhbmEucmhlZUBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0ppOE9KZjU2c1lkNWQ0Y2NtTGhSeGNHcDhmREp6RXl0N2VKZ2x5X1FpVDIwNGFnZz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTkzODQ4NjA2NH0.RNsqAZ7V_4-M9iuJTSr_Hr7KECl4dbFnDENFZZAZIS4')
    c.fetch_version()

def get_LC10_ids_Sten2021():
    """
    Returns hard-coded LC10 neuron IDs from Sten2021 supplementary data.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['LC10a', 'LC10b', 'LC10c', 'LC10d']
                     containing LC10 neuron IDs from Sten2021 study.
    """
    # Hard-coded LC10 IDs based on Sten2021 supplementary table
    # This replaces the file-dependent manual_tab2csv function
    lc10_data = [
        [1110566098.0, 1111243870.0, 1111917081.0, 1111580845.0],
        [1110535962.0, 1111217863.0, 1110885932.0, 1110562345.0],
        [1110186452.0, 1136399017.0, 1140547788.0, 1110557531.0],
        [1110903013.0, 1136740119.0, 1142598324.0, 1110894595.0],
        [1110903093.0, 1141937695.0, 1167433954.0, 1110885663.0],
        [1110566383.0, 1141601145.0, 1167092919.0, 1137085385.0],
        [1136744598.0, 1142257556.0, 1167433990.0, 1137763222.0],
        [1139153136.0, 1166756335.0, 1136744615.0, 1138117141.0],
        [1138807924.0, 1136744594.0, 1136744508.0, 1049510640.0],
        [1079527389.0, 1136744536.0, 1139153107.0, 1079522281.0],
        [1048492504.0, 1049511270.0, 1138812529.0, 1079527169.0],
        [1048125254.0, 1049502699.0, 1137085556.0, 1048492326.0],
        [1049169888.0, 1048491559.0, 1049506374.0, 1048120973.0],
        [1048837074.0, 1048837669.0, 1077792387.0, 1049174217.0],
        [1080541803.0, 1080545525.0, 1080200318.0, 1080537608.0],
        [1079872487.0, 1080213108.0, 5812997245.0, 1080541849.0],
        [1080200419.0, 1080212848.0, 5812996568.0, 1109194016.0],
        [5812987709.0, 1080519686.0, 5812992408.0, 1080878335.0],
        [5812983845.0, 5812987740.0, 5813016076.0, 1080196056.0],
        [5813056280.0, 5813016661.0, 1876105544.0, 1079855015.0],
        [5813044569.0, 5813016381.0, 1940557189.0, 5812990315.0],
        [5813016380.0, 1200182520.0, 1719250390.0, 1935821486.0],
        [5813035545.0, 893287523.0, 1719591388.0, 5813048880.0],
        [1198123389.0, 893645771.0, 1719237849.0, 5813048452.0],
        [1199496243.0, 925008566.0, 5812986560.0, 5813016652.0],
        [892950729.0, 925017453.0, 2154767014.0, 5813041453.0],
        [892963919.0, 925017553.0, 2185481747.0, 5813041801.0],
        [892963332.0, 923977051.0, 5813131327.0, 5813031440.0],
        [924348411.0, 862610554.0, 5901214649.0, 1169492931.0],
        [924002690.0, 987091864.0, 5813058307.0, 1198468674.0],
        [862252074.0, 955383212.0, 5813027631.0, 1352598905.0],
        [862265699.0, 956043578.0, 5813016650.0, 1321222721.0],
        [861574640.0, 956056543.0, 5813035500.0, 1439927802.0],
        [892273269.0, 956060536.0, 1261902687.0, 893642012.0],
        [1016749259.0, 5813087952.0, 1226084614.0, 892623020.0],
        [1016749264.0, 1261902646.0, 1228817231.0, 892959608.0],
        [1016758041.0, 1200523558.0, 1172623224.0, 892609615.0],
        [986763734.0, 1536068485.0, 1198132022.0, 892963477.0],
        [986767293.0, 1563372679.0, 1533040839.0, 924344128.0],
        [986758977.0, 1533040625.0, 1657516638.0, 924344089.0],
        [986763451.0, 1347470894.0, 1622740353.0, 924689450.0],
        [987104744.0, 1442314421.0, 1625804774.0, 923644961.0],
        [987116716.0, 1411262721.0, 1383633527.0, 923661510.0],
        [1047784113.0, 955383233.0, 1379549812.0, 892268510.0],
        [1047784012.0, None, 1312368829.0, 892268119.0],
        [1017793356.0, None, 1315435360.0, 891918572.0],
        [1017797863.0, None, 1347125913.0, 891931735.0],
        [1017789715.0, None, 1346465519.0, 891936249.0],
        [1017802759.0, None, 1471600919.0, 987424258.0],
        [1018472039.0, None, 1444006860.0, 1017094185.0],
        [1017802884.0, None, 1438563366.0, 1017448980.0],
        [955029299.0, None, 924681085.0, 1017781118.0],
        [986059699.0, None, 923640363.0, 1016753670.0],
        [956082289.0, None, 861583357.0, 986759122.0],
        [985036637.0, None, 892277589.0, 1047119615.0],
        [985377724.0, None, 891595069.0, 1047080740.0],
        [985369220.0, None, 1016408310.0, 1046734991.0],
        [1048125065.0, None, 1017090133.0, 955374862.0],
        [986396100.0, None, 1018476201.0, 955365826.0],
        [None, None, 1018462968.0, 955374672.0],
        [None, None, 954675120.0, 955715991.0],
        [None, None, 1136399018.0, 954342682.0],
        [None, None, 1142266328.0, 954692704.0],
        [None, None, 1080532771.0, 955365353.0],
        [None, None, 1200182517.0, 1080200438.0],
        [None, None, 1017785388.0, 5813015763.0],
        [None, None, None, 5813048451.0],
        [None, None, None, 5813079014.0],
        [None, None, None, 893305416.0],
        [None, None, None, 923321015.0],
        [None, None, None, 923636089.0],
        [None, None, None, 892964102.0],
        [None, None, None, 923994401.0],
        [None, None, None, 861229045.0],
        [None, None, None, 955378609.0],
        [None, None, None, 954010798.0],
        [None, None, None, 955024729.0],
        [None, None, None, 986059778.0],
        [None, None, None, 986063833.0],
    ]
    
    cols = ['LC10a', 'LC10b', 'LC10c', 'LC10d']
    LC10_ids = pd.DataFrame(lc10_data, columns=cols)
    
    return LC10_ids


def manual_tab2csv(
    csv_fpath=None,
    supptable_fpath = '/Users/julianarhee/Documents/rutalab/projects/connectome/supptable2_Sten2021.rtf',
    create_new=False):
    """
    DEPRECATED: Use get_LC10_ids_Sten2021() instead.
    This function is kept for backward compatibility.
    """
    #supptable_fpath = '/Users/julianarhee/Documents/rutalab/projects/connectome/supptable2.rtf'
    lc_info_dir = os.path.split(supptable_fpath)[0]
    # output path
    if csv_fpath is None:
        csv_fpath = os.path.join(lc_info_dir, 'LC10_IDs.csv')

    import re

    try:
        LC10_ids = pd.read_csv(csv_fpath, index_col=0)
    except Exception as e:
        create_New=True

    if create_new:
        # read tabdata, and fix weird funky stuff
        tabdata = list(filter(None, [re.split('\s+', i.strip('\n')) for i in open(supptable_fpath)]))
        tdata = tabdata[8:]
        tdata[0] = tdata[0][2:]
    
        # aggregate into DF
        cols = ['LC10a', 'LC10b', 'LC10c', 'LC10d']
        tvals = [[int(re.sub("[^0-9]", "", v)) for v in l] for l in tdata]
        d_list=[]
        for li, lv in enumerate(tvals):
            if len(lv) ==3:
                d_ = pd.DataFrame([lv[0], None, lv[1], lv[2]], index=cols)
            elif len(lv) == 2:
                d_ = pd.DataFrame([None, None, lv[0], lv[1]], index=cols)
            elif len(lv)==1: # reconsturct longer 4th col
                d_ = pd.DataFrame([None, None, None, lv[0]], index=cols)
            else:
                d_ = pd.DataFrame(lv, index=cols)
            d_list.append(d_)
        LC10_ids = pd.concat(d_list, axis=1).T.reset_index(drop=True)
        
        # save
        LC10_ids.to_csv(csv_fpath)

    return LC10_ids

# From sandbox

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

#%%

def CoM(df, xvar='pos_x', yvar='pos_y', zvar=None, is_3d=False):
    '''
    Calculate center of mass for x, y coordinates in df.
    
    Arguments:
        df -- _description_

    Returns:
        _description_
    '''
    x = df[xvar].values
    y = df[yvar].values
    m = np.ones(df[xvar].shape)
    cgx = np.sum(x*m) / np.sum(m)
    cgy = np.sum(y*m) / np.sum(m)
    
    if is_3d:
        z = df[zvar].values
        cgz = np.sum(z*m) / np.sum(m)
        return cgz, cgy, cgz
    
    return cgx, cgy

#%%
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
