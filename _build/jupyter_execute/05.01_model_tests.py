#!/usr/bin/env python
# coding: utf-8

# # Model tests
# VMM:s will be developed for the reference ship using motion regression based on a series of model tests with a model that is free in six degrees of freedome. A summary of the available model tests is shown in {ref}`tab:df_runs_table`.

# In[1]:


# %load imports.py
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_kedro', '')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False  ## (To fix autocomplete)')

import pandas as pd
pd.set_option('display.max_columns', 500)
from src.models.vmm import ModelSimulator
import matplotlib.pyplot as plt
from src.visualization.plot import track_plots, plot, captive_plot
import kedro
import numpy as np
import os.path
import anyconfig

import matplotlib
matplotlib.rcParams["figure.figsize"] = (7,4)
plt.style.use('presentation')


from myst_nb import glue
from src.symbols import *
import src.symbols as symbols
from src.system_equations import *

from IPython.display import display, Math, Latex, Markdown
from sympy.physics.vector.printing import vpprint, vlatex

from src.parameters import df_parameters
p = df_parameters["symbol"]

# Read configs:
conf_path = os.path.join("../../conf/base/")
runs_globals_path = os.path.join(
    conf_path,
    "runs_globals.yml",
)

runs_globals = anyconfig.load(runs_globals_path)
model_test_ids = runs_globals["model_test_ids"]

join_globals_path = os.path.join(
    conf_path,
    "join_globals.yml",
)

joins = runs_globals["joins"]
join_runs_dict = anyconfig.load(join_globals_path)

globals_path = os.path.join(
    conf_path,
    "globals.yml",
)
global_variables = anyconfig.load(globals_path)



vmms = global_variables["vmms"]
only_joined = global_variables[
    "only_joined"
]  # (regress/predict with only models from joined runs)S


# In[2]:


ship_data = catalog.load("ship_data")

#from wPCC_pipeline.pipelines.preprocess.nodes import track_plot
from src.visualization.plot import track_plots, track_plot, plot


# In[3]:


dataframes = {}
df = pd.DataFrame()

for id in model_test_ids:
    
    df_ = catalog.load(f"{ id }.raw_data")
    df_['psi+'] = df_['psi'] + np.deg2rad(90)
    df_['-y0'] = -df_['y0']
    df_['delta_deg'] = np.rad2deg(df_['delta'])
    
    dataframes[id] = df_


# In[4]:


df_runs = catalog.load("runs_meta_data")
df_runs.index = df_runs.index.astype('str')
df_runs = df_runs.loc[model_test_ids].copy()

mask = df_runs['test_type'] == 'rodergrundvinkel'
df_runs.loc[mask,'test_type'] = 'yaw rate'
mask = df_runs['test_type'] != 'zigzag'
df_runs.loc[mask,'comment'] = np.NaN
mask = ((df_runs['comment'].notnull()) & (df_runs['test_type'] == 'zigzag'))
df_runs['angle'] = df_runs.loc[mask,'comment'].apply(lambda x:int(x[3:5]))
df_runs['direction'] = df_runs.loc[mask,'comment'].apply(lambda x:x[8:11])

df_runs.sort_values(by=['test_type','ship_speed','angle'], inplace=True)

df_runs_table = df_runs.rename(columns={'ship_speed':'Initial speed [m/s]','test_type':'type'})
df_runs_table = df_runs_table[['Initial speed [m/s]','type','angle','direction']]

formatter={'Initial speed [m/s]' : "{:.2f}", 'angle' : "{:.0f}"}

df_runs_table = df_runs_table.style.format(formatter=formatter, na_rep='')

glue("df_runs_table", df_runs_table)


# ```{glue:figure} df_runs_table
# :name: "tab:df_runs_table"
# 
# Model tests
# ```

# In[5]:


for test_type, df_ in df_runs.groupby(by=['test_type']):
    
    dataframes_ = {key:value for key,value in dataframes.items() if key in df_.index}
    
    if test_type == 'reference speed':
        continue
    
    fig = track_plots(dataframes=dataframes_, lpp=ship_data['L'], beam=ship_data['B'], x_dataset='-y0',
    y_dataset='x0', psi_dataset='psi+', plot_boats=True, N=7)
    ax = fig.axes
    ax.set_title(f"{test_type}")
    ax.get_legend().set_visible(False)


# In[ ]:




