#!/usr/bin/env python
# coding: utf-8

# # Vessel Manoeuvring Models
# Many simulation model for ship manoeuvring have been developed in the field of ship hydrodynamics such as: the Abkowitz model {cite:p}`abkowitz_ship_1964` or the Norrbin model {cite:p}`norrbin_study_1960`.
# This chapter will develop a general simulation model for ship manoeuvring, that can be further specified to become either the Abkowitz or Norbin model. Expressing the models on a general form is important in this research where many different models will be tested and compared.

# In[1]:


# %load imports.py
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_kedro', '')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False  ## (To fix autocomplete)')

import pandas as pd
from src.models.vmm import ModelSimulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (10,10)
plt.style.use('presentation')
from src.visualization.plot import track_plots, plot, captive_plot
import kedro
import numpy as np
import os.path
import anyconfig


from myst_nb import glue
from src.symbols import *
import src.symbols as symbols
from src.system_equations import *

from IPython.display import display, Math, Latex, Markdown
from sympy.physics.vector.printing import vpprint, vlatex

from src.models.regression import MotionRegression

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



vmm_names = global_variables["vmms"]
only_joined = global_variables[
    "only_joined"
]  # (regress/predict with only models from joined runs)S

vmms = {}
for vmm_name in vmm_names:
    vmms[vmm_name] = catalog.load(vmm_name)


# 3DOF system for manoeurving:

# In[2]:


eq_system


# The manoeuvring simulation can now be conducted by numerical integration of the above equation. The main difference between various vessel manoeuvring models such as the Abkowitz model {cite:p}`abkowitz_ship_1964` or the Norrbin model {cite:p}`norrbin_study_1960` lies in how the hydrodynamic functions $X_D(u,v,r,\delta,thrust)$, $Y_D(u,v,r,\delta,thrust)$, $N_D(u,v,r,\delta,thrust)$ are defined. These functions cane be found in [Appendix](appendix_vmms.md).

# Note that a coefficient $X_{thrust}$ has been added to the Abkowitz X equation to allow for propeller thrust as an input to the model. 

# In[3]:


vmms['vmm_abkowitz'].Y_qs_eq


# In[4]:


vmms['vmm_linear'].Y_qs_eq


# This equation can be rewritten to get the acceleration on the left hand side:

# In[5]:


eq_acceleration_matrix_clean


# where $S$ is a helper variable:

# In[6]:


eq_S


# A state space model for manoeuvring can now be defined with six states:

# In[7]:


eq_x


# An transition function $f$ defines how the states changes with time:

# In[8]:


eq_state_space


# Using geometrical relations for how $x_0$, $y_0$ and $\Psi$ depend on $u$, $v$, and $r$ and the time derivatives that was derived above: $\dot{u}$, $\dot{v}$, $\dot{r}$, the transition function can be written:

# In[9]:


eq_f


# In[10]:


vmm_name = 'vmm_martins_simple'
ek = catalog.load(f"{ vmm_name }.ek")
model = catalog.load(f"{ vmm_name}.motion_regression.joined.model")
vmm = catalog.load(f"{ vmm_name }")
ek.parameters = model.parameters
added_masses = catalog.load("added_masses")


# In[11]:


t = np.arange(0, 70, 0.01)
input_columns = ['delta','U','thrust']
state_columns = ['x0', 'y0', 'psi', 'u', 'v', 'r']
data = pd.DataFrame(index=t, columns=state_columns + input_columns)
data['u'] = 2
data['delta'] = np.deg2rad(15)
data['thrust'] = 30
data.fillna(0, inplace=True)
data['U'] = np.sqrt(data['u']**2 + data['v']**2)


result = model.simulate(df_=data)

dataframes = {'simulation': result.result}
#dataframes['simulate'] = ek.simulate(data=data, input_columns=input_columns, solver='Radau')
track_plots(dataframes, lpp=model.ship_parameters['L'], beam=model.ship_parameters['B'], N=15, 
            styles={'simulation':{'alpha':1}});
result.result.to_csv('example.csv')


# In[ ]:





# In[ ]:




