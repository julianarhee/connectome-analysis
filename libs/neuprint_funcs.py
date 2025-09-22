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
import neuprint as neu

from sklearn import mixture


from neuprint import Client
import numpy as np
# c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imp1bGlhbmEucmhlZUBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0ppOE9KZjU2c1lkNWQ0Y2NtTGhSeGNHcDhmREp6RXl0N2VKZ2x5X1FpVDIwNGFnZz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg5NzQzNDg4MX0.Idd57ywnGRIdIGfrVXVguCN7AWFafq3LizEDdBY45Co')
# c.fetch_version()

def flatten(xss):
    return [x for xs in xss for x in xs]

#%%
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


