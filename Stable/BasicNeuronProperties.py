# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:49:00 2023

@author: dowel
"""
import numpy as np
#%%
class neuron_properties:
    def __init__(self):
        self.name = 'This'
    def MBON_compartment_valence(self):
        MBON_names = np.array(['MBON01', 'MBON02', 'MBON03', 'MBON04', 'MBON05', 'MBON06',
               'MBON07', 'MBON08', 'MBON09', 'MBON10', 'MBON11', 'MBON12', 'MBON13',
               'MBON14', 'MBON15', 'MBON15-like', 'MBON16', 'MBON17',
               'MBON17-like', 'MBON18', 'MBON19', 'MBON20', 'MBON21', 'MBON22',
               'MBON23', 'MBON24', 'MBON25', 'MBON26', 'MBON27', 'MBON28',
               'MBON29', 'MBON30', 'MBON31', 'MBON32', 'MBON33', 'MBON34',
               'MBON35'])
        MBON_compartments = [["g5","b'2"],
                             ["b'2","b2"],
                             ["b'2"],
                             ["b'2"],
                             ["g4"],
                             ["b1"],
                             ["a1"],
                             ["g3"],
                             ["g3","b'1"],
                             ["b'1"],
                             ["g1","ped"],
                             ["g2","a'1"],
                             ["a'2"],
                             ["a3"],
                             ["a'1"],
                             ["a'1"],
                             ["a'3"],
                             ["a'3"],
                             ["a'3"],
                             ["a2"],
                             ["a2","a3"],
                             ["g1","g2"],
                             ["g4","g5"],
                             ["calyx"],
                             ["a2"],
                             ["b2","g5"],
                             ["g1","g2"],
                             ["b'2"],
                             ["g5"],
                             ["a'3"],
                             ["g4","g5"],
                             ["g1","g2","g3"],
                             ["a'1"],
                             ["g2"],
                             ["g2","g3"],
                             ["g2"],
                             ["g2"]]
        
        MBON_compartments_cat = np.array(["g5 b'2",
                             "b'2 b2",
                             "b'2",
                             "b'2",
                             "g4",
                             "b1",
                             "a1",
                             "g3",
                             "g3 b'1",
                             "b'1",
                             "g1 ped",
                             "g2 a'1",
                             "a'2",
                             "a3",
                             "a'1",
                             "a'1",
                             "a'3",
                             "a'3",
                             "a'3",
                             "a2",
                             "a2 a3",
                             "g1 g2",
                             "g4 g5",
                             "calyx",
                             "a2",
                             "b2 g5",
                             "g1 g2",
                             "b'2",
                             "g5",
                             "a'3",
                             "g4 g5",
                             "g1 g2 g3",
                             "a'1",
                             "g2",
                             "g2 g3",
                             "g2",
                             "g2"])
        Valence = np.array([-1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   -1,
                   0,
                   1,
                   -1,
                   1,
                   -1,
                   -1,
                   1,
                   -1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1])
        Plume = np.array([-1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   -1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   -1,
                   0,
                   1,
                   -1,
                   1,
                   -1,
                   -1,
                   1,
                   -1,
                   1,
                   1,
                   1,
                   1,
                   1,
                   1])
        
        self.MBON_dict = {'MBONs': MBON_names,'Compartments': MBON_compartments,
                          'Compartments_cat': MBON_compartments_cat,
                          'Valence': Valence,'Plume': Plume}
        return self
        