# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:19:01 2023

@author: dowel
"""

#%% Aim of this analysis is to predict in rough terms the effect of differential
# compartment activity from MBONs on tangential neurons and post-synaptic cells
# in FB

# Assumption 1
# Network is non spiking and each unit has a linear-input-output
# Assumption 2
# Network is restricted to direct MBON connections - could expand
# Assumption 3
# DAN activity correlates with decreased activity in each compartment
# Assumption 4
# Tangential neurons are all inhibitory unless specified otherwise
#%% Import modules
from neuprint import Client
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
import pandas as pd
import numpy as np
from neuprint import fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
from neuprint import queries 
from scipy import stats
import matplotlib.pyplot as plt
#%% MBON dictionary of properties
def set_MBON_dicts():
    # Easier to set manually
    # All MBONs in neuprint: is missing MBON08
    mbon_names = np.array(['MBON01', 'MBON02', 'MBON03', 'MBON04', 'MBON05', 'MBON06',
           'MBON07', 'MBON09', 'MBON10', 'MBON11', 'MBON12', 'MBON13',
           'MBON14', 'MBON15', 'MBON15-like', 'MBON16', 'MBON17',
           'MBON17-like', 'MBON18', 'MBON19', 'MBON20', 'MBON21', 'MBON22',
           'MBON23', 'MBON24', 'MBON25', 'MBON26', 'MBON27', 'MBON28',
           'MBON29', 'MBON30', 'MBON31', 'MBON32', 'MBON33', 'MBON34',
           'MBON35'], dtype=object)
    
    # Neurotransmitter type
    ACh_MBONs = np.array(['MBON24','MBON33','MBON35','MBON29','MBON27','MBON24'
                          'MBON26','MBON28','MBON23','MBON18','MBON19','MBON14',
                         'MBON12','MBON21','MBON15','MBON13','MBON16','MBON17'],dtype=object)
    Glu_MBONs = np.array(['MBON25','MBON30','MBON34','MBON07','MBON06','MBON02','MBON05',
                          'MBON21','MBON03','MBON04'],dtype = object)
    GABA_MBONs = np.array(['MBON20','MBON32','MBON10','MBON31','MBON11','MBON08','MBON09'],dtype=object)
    
    compartments = np.array(["a'1", "a'2", "a'3", "a'L", 'a1', 'a2', 'a3',  'aL', "b'1", "b'2", 
     "b'L", 'b1', 'b2', 'bL', 'g1', 'g2', 'g3', 'g4', 'g5', 'gL'],dtype=object)
      
    comp_array = np.zeros([len(mbon_names), len(compartments)])
    nt_sign = np.ones(len(mbon_names))
    criteria = NC(type='MBON.*')
    df_neuron,roi_neuron = fetch_neurons(criteria)
    in_roi = pd.Series.to_numpy(df_neuron['inputRois'])
    type_list = pd.Series.to_numpy(df_neuron['type'])
    # Iterate through MBONs get comparments and build NT arrays and compartment arrays
    for i,n in enumerate(mbon_names):
        indx = type_list==n
        
        # Get compartments and assign into compartment matrix
        this_in = in_roi[indx]
        in_all = []
        for tl in this_in:
            in_all.extend(tl)
        in_all = np.unique(in_all)
        
        
        
        for ic, c in enumerate(compartments):
            for r in in_all:
                if c in r:
                    comp_array[i,ic] = 1
                
        # Assign NT sign to neurons
        glu_dx = Glu_MBONs == n
        gaba_dx = GABA_MBONs==n
        if sum(glu_dx)>0 or sum(gaba_dx)>0:
            nt_sign[i] = -1
               
    MBON_dicts = dict({'Neur_names':mbon_names, 'NT_sign': nt_sign, 'Comp_names': compartments, 
                       'Comp_array': comp_array})
    return MBON_dicts
    
    # 
MBON_dicts = set_MBON_dicts()
#%% 
import pyarrow as pa
import pyarrow.feather as feather
import bz2

def get_NT_identity(body_ids):
    print('Loading NT dataset')
    filepath = 'C:\\Users\\dowel\\Documents\\PostDoc\\ConnectomeMining\\hemibrain-v1.2-tbar-neurotransmitters.feather.bz2'

    with bz2.open(filepath, 'rb') as bz2_file:
        # Read the decompressed binary data
        decompressed_data = bz2_file.read()

    # Read the Feather data from the decompressed binary data
    df = feather.read_feather(pa.py_buffer(decompressed_data))
    print('Loaded')
    df_names = df.head(0)
    nt_list = [i for i in df_names if 'nts_8' in i]
    NT_matrix = np.zeros([len(body_ids),len(nt_list)])
    syn_body_ids = pd.Series.to_numpy(df['body'])
    for i, bod in enumerate(body_ids):
        bodx = syn_body_ids==bod
        for n, nt in enumerate(nt_list):
            t_nt = df[nt][bodx]
           
            NT_matrix[i,n] = np.sum(t_nt)
    nt_id = np.argmax(NT_matrix,1)
    print('Got NTs')
    # NT list 0: gaba 1: ACh 2: Glu 3: 5HT 4: octopamine 5: DA 6: neither
    return NT_matrix, nt_id
#%% Return tangential neuron dictionary of properties
def set_neur_dicts(types):
    print('Getting neurons')
    criteria = NC(type=types)
    df_neuron,roi_neuron = fetch_neurons(criteria)
    print('Getting NTs')
    NT_matrix, nt_id = get_NT_identity(df_neuron['bodyId'])
    n_names = pd.Series.to_numpy(df_neuron['type'])
    tan_names = np.unique(n_names)
    nt_sign = np.empty(np.shape(tan_names))
    nt_ids = np.empty(np.shape(tan_names))
    nt_sign_index = [-1, 1, -1, 0, 1, -1, 0]# zero for 5HT, 1 for octopamine, -1 for DA
    for i,t in enumerate(tan_names):
        bod_dx = n_names==t
        n_types = nt_id[bod_dx]
        nt_type = stats.mode(n_types)
        nt_sign[i] = nt_sign_index[nt_type.mode]
        nt_ids[i] = nt_type.mode
        
    NT_list = ['gaba', 'ACh', 'Glu', '5HT', 'Oct',  'DA', 'NA']
    neur_dicts = dict({'Neur_names':tan_names, 'NT_sign':nt_sign, 'NT_id':nt_ids, 'NT_list': NT_list})
    print('Done')
    return neur_dicts

#%% 

def run_sim_act(inputs,neurons,iterations):
    neur_dicts = set_neur_dicts(neurons) #gets types and associated NTs
    n_ids = np.empty(0,dtype='int64')
    n_types = np.empty(0)
    for n in neurons:
        criteria = NC(type=n)
        df_neuron,roi_neuron = fetch_neurons(criteria)    
        n_ids = np.append(n_ids,pd.Series.to_numpy(df_neuron['bodyId'],dtype='int64'))
        n_types = np.append(n_types,df_neuron['type'])
    criteria = NC(bodyId=n_ids)
    
    neurons_df, roi_con_df = fetch_adjacencies(sources = n_ids,targets = n_ids,batch_size=200)#get connections
    
    
    
    neuron_basic,roi_counts = fetch_neurons(n_ids)
    
    norm_dx = np.in1d(neuron_basic['bodyId'],neurons_df['bodyId'])
    
    norm_outweight = neuron_basic['post'][norm_dx] # There is also an upstream field that I think includes connections to unidentified neurites
    norm_outweight = np.expand_dims(norm_outweight,1)
    norm_outweight = np.transpose(norm_outweight)
    
    # Set up connectivity matrix - 
    conmatrix = np.zeros([len(neurons_df), len(neurons_df)])
    bod_id = pd.Series.to_numpy(neurons_df['bodyId'])
    con_in = pd.Series.to_numpy(roi_con_df['bodyId_pre'])
    con_out = pd.Series.to_numpy(roi_con_df['bodyId_post'])
    weights = pd.Series.to_numpy(roi_con_df['weight'])
    types = pd.Series.to_numpy(neurons_df['type'])
    for i,b in enumerate(bod_id):   
        indx = con_in==b
        t_outs = con_out[indx]
        t_w = weights[indx]
        
        t_outs_u = np.unique(t_outs) # annoyingly not all connections are summed as 1-2-1
        t_w_u = np.zeros_like(t_outs_u)
        for iu, u in enumerate(t_outs_u):
            todx = t_outs==u
            t_w_u[iu] = np.sum(t_w[todx])
        
        t_out_dx = np.where(np.in1d(bod_id,t_outs_u))
        conmatrix[i,t_out_dx] = t_w_u
    
    conmatrix = conmatrix/norm_outweight # normalise input strength by total input to each neuron
    # Set input weights
    w_types = neur_dicts['Neur_names']
    for r, i in enumerate(w_types):
        t_sign = neur_dicts['NT_sign'][r]
        tdx = types==i
        conmatrix[tdx,:] = conmatrix[tdx,:]*t_sign
        
    # Set up activation vector
    a_vec = np.zeros([len(types), 1])
    for i in inputs:
        idx = types==i
        a_vec[idx] = 1
    
    
    act_vec = np.transpose(a_vec)
    activity_mat = act_vec
    act_vec = np.matmul(act_vec,conmatrix)
    activity_mat = np.append(activity_mat,act_vec,axis=0)
    for i in range(iterations-1):
        act_vec[act_vec<0] = 0 # need to stop negative activity from being propagated, this effectively puts a relu in place
        act_vec = np.matmul(act_vec,conmatrix)
        activity_mat = np.append(activity_mat,act_vec,axis=0)
        
    
    u_types = np.unique(types)
    activity_mat_type = np.zeros([iterations+1,len(u_types),])
    for i, t in enumerate(u_types):
        tdx = types==t
        activity_mat_type[:,i] = np.mean(activity_mat[:,tdx],axis=1)
    
    sim_output = dict({'ActivityAll': activity_mat, 'TypesAll': types, 'ROI_ID': bod_id, 
                       'MeanActivityType': activity_mat_type, 'TypesSmall': u_types})    
    
    return sim_output

#%% 
def x_tick_colours(xt,xtlabs,types,typecolours,**kwargs):
    plt.xticks(xt,labels=xtlabs,**kwargs)
    ax = plt.subplot()
    clist = np.zeros([len(xt),3])
    for i, t in enumerate(types):
        tdx = [i for i, it in enumerate(xtlabs) if t in it]
        clist[tdx,:] = typecolours[i,:]
        
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(clist[i,:])
        
    # Function will output different coloured
#%% MBON 1
sim_output = run_sim_act(['MBON01','FC2B'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)

tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.001,vmin=-0.001,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% 21
sim_output = run_sim_act(['MBON21','FC2B'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.001,vmin=-0.001,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% Gamma 4/5
sim_output = run_sim_act(['MBON21','MBON05','MBON01','MBON29','MBON27','MBON24'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% Gamma 3
sim_output = run_sim_act(['MBON30','MBON33','MBON08','MBON09'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% MBON 30
sim_output = run_sim_act(['MBON30'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)

#%% FB5AB activation
sim_output = run_sim_act(['FB5AB'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%%
sim_output = run_sim_act(['FB4R'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% Compare gamma3 with gamma 4/5
sim_output_g45 = run_sim_act(['MBON21','MBON05','MBON01','MBON29','MBON27','MBON24'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
sim_output_g3 = run_sim_act(['MBON30','MBON33','MBON08','MBON09'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
sim_output_g23 = run_sim_act(['MBON30','MBON33','MBON08','MBON09','MBON12','MBON35','MBON32','MBON34','MBON20','MBON25'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
x = sim_output_g45['MeanActivityType'][1,:]
y = sim_output_g3['MeanActivityType'][1,:]
#%% gamma 4/5, b1, b1' and b2 and b2'
sim_output_g45_plus = run_sim_act(['MBON21','MBON05','MBON01','MBON29','MBON27','MBON24','MBON10','MBON26','MBON06','MBON02'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% hDeltaC
sim_output = run_sim_act(['hDeltaC'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],3)
tsim = sim_output['MeanActivityType']
plt.imshow(tsim,vmax=0.01,vmin=-0.01,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
#%% 
tan_dict = set_neur_dicts('FB.*')
FC_dict = set_neur_dicts('FC.*')
#%%
tan_dx = [i for i,t in enumerate(sim_output_g3['TypesSmall']) if 'FB' in t]
t_x = x[tan_dx]
t_y = y[tan_dx]
tan_names = sim_output_g3['TypesSmall'][tan_dx]
plt.Figure()
plt.scatter(t_x,t_y,c= [0, 0, 0])
large_tan = np.where(np.logical_or(np.abs(t_x)>0.005,np.abs(t_y)>0.005))
colours = np.array([[77,175,74],
                     [255,127,0],
                     [55,126,184],
                     [247,129,191],
                     [166,86,40],
                     [152,78,163],
                     [153,153,153]
                     ])/255
for i in large_tan[0]:
    tan_dict_dx = tan_dict['Neur_names']==tan_names[i]
    t_nt = tan_dict['NT_id'][tan_dict_dx]
    t_nt = int(t_nt[0])
    nt_id = tan_dict['NT_list'][t_nt]
    print(tan_names[i] + ' ' + nt_id)
    
    plt.scatter(t_x[i],t_y[i],c=colours[t_nt,:])
    plt.text(t_x[i],t_y[i],tan_names[i])

plt.xlabel('Gamma 4/5 activation')
plt.ylabel('Gamma 3 activation')
#%% get tangential neuron properties

mbd = set_neur_dicts('MBON.*')
#%% Gamma 3 activation
tsim = sim_output_g3['MeanActivityType']
ondx = np.max(np.abs(tsim[:2,:]),axis=0)>0.001

plt.figure(figsize=(12,4))
plt.imshow(tsim[:2,ondx],vmax=0.005,vmin=-0.005,aspect='auto',interpolation='none',cmap='Greys_r')
t_names = sim_output_g45['TypesSmall']
plt.xticks(np.linspace(0,len(t_names[ondx])-1,len(t_names[ondx])),labels= t_names[ondx],rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.show()
#%% Gamma 4/5 beta1-2 beta'1-2
savedir = 'C:\\Users\\dowel\\Documents\\PostDoc\\ConnectomeMining\\ActivationSimulations\\'
savename = 'Gamma45_plus'
non_mbon = [i for i,t in enumerate(sim_output_g3['TypesSmall']) if 'MBON' not in t]
tsim = sim_output_g45_plus['MeanActivityType']
ts_norm = tsim[:,non_mbon]
t_norm = np.max(np.abs(ts_norm),axis=1)
t_norm[0] = 1
t_norm = t_norm.reshape(-1,1)
tsim = tsim/t_norm

ondx = np.max(np.abs(tsim[:3,:]),axis=0)>0.01
plt.figure(figsize=(16,4))
plt.imshow(tsim[:3,:],vmax=.5,vmin=-.5,aspect='auto',interpolation='none',cmap='coolwarm')
t_names = sim_output_g45['TypesSmall']
#plt.xticks(np.linspace(0,len(t_names[ondx])-1,len(t_names[ondx])),labels= t_names[ondx],rotation=90,fontsize=8)
plt.yticks(np.linspace(0,2,3))
plt.ylabel('Iterations')
plt.subplots_adjust(bottom=0.4)
colours = np.array([[27,158,119],
[217,95,2],
[117,112,179],
[231,41,138],
[102,166,30],
[230,171,2]])/255
x_tick_colours(np.linspace(0,len(t_names[:])-1,len(t_names[:])),t_names[:],['FB','MBON','FC','hDelta','vDelta','PFL'],
               colours,rotation=90,fontsize=8)


plt.savefig(savedir +savename +'.eps')
plt.show()
#%% Gamma 2-3 activation
savedir = 'C:\\Users\\dowel\\Documents\\PostDoc\\ConnectomeMining\\ActivationSimulations\\'
savename = 'Gamma2_3'
tsim = sim_output_g23['MeanActivityType']
ts_norm = tsim[:,non_mbon]
t_norm = np.max(np.abs(ts_norm),axis=1)
t_norm[0] = 1
t_norm = t_norm.reshape(-1,1)
tsim = tsim/t_norm

ondx = np.max(np.abs(tsim[:3,:]),axis=0)>0.025
plt.figure(figsize=(16,4))
plt.imshow(tsim[:3,:],vmax=.5,vmin=-.5,aspect='auto',interpolation='none',cmap='coolwarm')
t_names = sim_output_g45['TypesSmall']
#plt.xticks(np.linspace(0,len(t_names[ondx])-1,len(t_names[ondx])),labels= t_names[ondx],rotation=90,fontsize=8)
plt.yticks(np.linspace(0,2,3))
plt.ylabel('Iterations')
plt.subplots_adjust(bottom=0.4)

colours = np.array([[27,158,119],
[217,95,2],
[117,112,179],
[231,41,138],
[102,166,30],
[230,171,2]])/255
x_tick_colours(np.linspace(0,len(t_names[:])-1,len(t_names[:])),t_names[:],['FB','MBON','FC','hDelta','vDelta','PFL'],
               colours,rotation=90,fontsize=6)

plt.savefig(savedir +savename +'.eps')
plt.show()
#%% Plot 1st iteration against one another
savedir = 'C:\\Users\\dowel\\Documents\\PostDoc\\ConnectomeMining\\ActivationSimulations\\'
savename = '1st Iteration differences'

tsim = sim_output_g23['MeanActivityType']
ts_norm = tsim[:,non_mbon]
tan_names = sim_output_g23['TypesSmall'][non_mbon]
t_norm = np.max(np.abs(ts_norm),axis=1)
t_norm[0] = 1
t_norm = t_norm.reshape(-1,1)
tsim_g23 = tsim/t_norm

tsim = sim_output_g45_plus['MeanActivityType']
ts_norm = tsim[:,non_mbon]
t_norm = np.max(np.abs(ts_norm),axis=1)
t_norm[0] = 1
t_norm = t_norm.reshape(-1,1)
tsim_g45plus = tsim/t_norm
x = tsim_g45plus[1,non_mbon]
y = tsim_g23[1,non_mbon]
plt.Figure()

plt.plot([-1,1],[0,0],c='k',zorder =1)
plt.plot([0,0],[-1,1],c='k',zorder=1)
plt.scatter(x,y,zorder=2)
plt.xlim([-.25,1.1])
plt.ylim([-1.1, .25])

ldx = np.where(np.logical_or(abs(x)>0.1, abs(y)>0.1))
for i in ldx[0]:

    plt.text(x[i],y[i],tan_names[i])


plt.xlabel('Norm activity Gamma 4 5 plus')
plt.ylabel('Norm activity Gamma 2 3')
plt.title('1st Iteration')

plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(savedir +savename +'.pdf', format='pdf')
plt.show()
#%% Plot 2nd iteration against one another
savedir = 'C:\\Users\\dowel\\Documents\\PostDoc\\ConnectomeMining\\ActivationSimulations\\'
savename = '2nd Iteration differences'


x = tsim_g45plus[2,non_mbon]
y = tsim_g23[2,non_mbon]
plt.Figure()
plt.plot([-1,1],[0,0],c='k',zorder =1)
plt.plot([0,0],[-1,1],c='k',zorder=1)
plt.scatter(x,y,zorder=2)
plt.xlim([-1.1,0.25])
plt.ylim([-1.1, 1.1])

ldx = np.where(np.logical_or(abs(x)>0.2, abs(y)>0.2))
for i in ldx[0]:
    if 'FB' in tan_names[i]:
        continue
    plt.text(x[i],y[i],tan_names[i],fontsize=8)


plt.xlabel('Norm activity Gamma 4 5 plus')
plt.ylabel('Norm activity Gamma 2 3')
plt.title('2nd Iteration')
plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(savedir +savename +'.pdf', format='pdf')
plt.show()

#%%
input_dict = set_MBON_dicts()
output_dict = set_neur_dicts('FB.*')
#%% 
from neuprint.utils import connection_table_to_matrix
from neuprint import merge_neuron_properties
neurons_df, roi_con_df = fetch_adjacencies(sources = input_dict['Neur_names'],targets = output_dict['Neur_names'])
conn_df = merge_neuron_properties(neurons_df, roi_con_df, ['type', 'instance'])
con_matrix = connection_table_to_matrix(roi_con_df, 'bodyId', sort_by='type')

#%% Get neurotransmitter identities of other neurons
neuron_df, conn_df = fetch_adjacencies(NC(type='Delta.*'), NC(type='PEN.*'))
conn_df.sort_values('weight', ascending=False)
conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
matrix = connection_table_to_matrix(conn_df, 'bodyId', sort_by='type')