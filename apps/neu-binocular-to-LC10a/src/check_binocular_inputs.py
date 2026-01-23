#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : check_binocular_inputs.py
Created        : 2025/10/25 14:53:59
Project        : /Users/julianarhee/Repositories/connectome-analysis/apps/neu-binocular-to-LC10a/src
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

from neuprint import Client
import neuprint as neu
from neuprint import NeuronCriteria as NC
from neuprint.utils import connection_table_to_matrix

import neuprint_funcs as npf

# %%
# Load token and get client
c = npf.load_token_and_connect()
c.fetch_version()
# %%
# Get all 