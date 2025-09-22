#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Mar 9 11:06:00 2024
@File    :   utils.py
@Time    :   2024/05/09 10:07:48
@Author  :   julianarhee 
'''

import os
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