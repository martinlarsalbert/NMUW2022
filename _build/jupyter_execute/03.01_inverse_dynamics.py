#!/usr/bin/env python
# coding: utf-8

# ## Inverse dynamics

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
matplotlib.rcParams["figure.figsize"] = (7,4)
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
from src.substitute_dynamic_symbols import run
from src.models.diff_eq_to_matrix import DiffEqToMatrix
p = df_parameters["symbol"]
import statsmodels.api as sm

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


# In[2]:


vmm_name = 'vmm_martins_simple'
vmm = vmms[vmm_name]
data = pd.read_csv('example.csv', index_col=0)
added_masses = catalog.load("added_masses")
model = catalog.load(f"{ vmm_name}.motion_regression.joined.model")

regression = MotionRegression(
        vmm=vmm,
        data=data,
        added_masses=added_masses,
        prime_system=model.prime_system,
        ship_parameters=model.ship_parameters,
        #exclude_parameters={"Xthrust": 1.0, "Ydelta": 1},
    )


# In[3]:


eq_system


# In[4]:


solution = sp.solve(eq_system.doit(),X_D,Y_D,N_D, dict=True)[0]

eq_XD = sp.Eq(X_D, solution[X_D])
eq_YD = sp.Eq(Y_D, solution[Y_D])
eq_ND = sp.Eq(N_D, solution[N_D])

display(eq_XD)
display(eq_YD)
display(eq_ND)


# In[5]:


display(vmm.X_qs_eq)
display(vmm.Y_qs_eq)
display(vmm.N_qs_eq)


# In[6]:


subs = [(value, key ) for key,value in p.items()]
subs.append((u1d,'u1d'))
subs.append((v1d,'v1d'))
subs.append((r1d,'r1d'))

eq = eq_XD.subs(subs)
lambda_X_D = sp.lambdify(list(eq.rhs.free_symbols), eq.rhs)

eq = eq_YD.subs(subs)
lambda_Y_D = sp.lambdify(list(eq.rhs.free_symbols), eq.rhs)

eq = eq_ND.subs(subs)
lambda_N_D = sp.lambdify(list(eq.rhs.free_symbols), eq.rhs)


# In[7]:


df_captive = data.copy()
df_captive_prime = model.prime_system.prime(df_captive, U=data['U'])

df_captive_prime['fx'] = run(lambda_X_D, 
                             inputs=df_captive_prime, 
                             **model.ship_parameters_prime, 
                             **added_masses)

df_captive_prime['fy'] = run(lambda_Y_D, 
                             inputs=df_captive_prime, 
                             **model.ship_parameters_prime, 
                             **added_masses)

df_captive_prime['mz'] = run(lambda_N_D, 
                             inputs=df_captive_prime, 
                             **model.ship_parameters_prime, 
                             **added_masses)


# In[8]:


Y_D_ = sp.symbols('Y_D')
eq = vmm.Y_qs_eq.subs(Y_D,Y_D_)
diff_eq_Y = DiffEqToMatrix(eq, label=Y_D_, base_features=[u,v,r,delta,thrust])

X_Y,y_Y = diff_eq_Y.calculate_features_and_label(data=df_captive_prime, y=df_captive_prime['fy'])

model_Y = sm.OLS(y_Y, X_Y)
result_Y = model_Y.fit()


# In[9]:


N_D_ = sp.symbols('N_D')
eq = vmm.N_qs_eq.subs(N_D,N_D_)
diff_eq_N = DiffEqToMatrix(eq, label=N_D_, base_features=[u,v,r,delta,thrust])

X_N,y_N = diff_eq_N.calculate_features_and_label(data=df_captive_prime, y=df_captive_prime['mz'])

model_N = sm.OLS(y_N, X_N)
result_N = model_N.fit()


# In[10]:


X_D_ = sp.symbols('X_D')
eq = vmm.X_qs_eq.subs(X_D,X_D_)
diff_eq_X = DiffEqToMatrix(eq, label=X_D_, base_features=[u,v,r,delta,thrust], exclude_parameters={'Xthrust':model.parameters['Xthrust']})

X_X,y_X = diff_eq_X.calculate_features_and_label(data=df_captive_prime, y=df_captive_prime['fx'])

model_X = sm.OLS(y_X, X_X)
result_X = model_X.fit()


# In[11]:


df_parameters_X = pd.DataFrame(pd.Series({key:value for key,value in model.parameters.items() if key[0]=='X' and value !=0}, name='real'))
df_parameters_X['regression'] = result_X.params
df_parameters_X.dropna(inplace=True)
df_parameters_X.index = p[df_parameters_X.index].apply(lambda x: "$%s$" % str(x).replace('delta',r'\delta'))
df_parameters_X.index.name = ''

df_parameters_Y = pd.DataFrame(pd.Series({key:value for key,value in model.parameters.items() if key[0]=='Y' and value !=0}, name='real'))
df_parameters_Y['regression'] = result_Y.params
df_parameters_Y.dropna(inplace=True)
df_parameters_Y.index = p[df_parameters_Y.index].apply(lambda x: "$%s$" % str(x).replace('delta',r'\delta').replace('thrust','T'))
df_parameters_Y.index.name = ''



df_parameters_N = pd.DataFrame(pd.Series({key:value for key,value in model.parameters.items() if key[0]=='N' and value !=0}, name='real'))
df_parameters_N['regression'] = result_N.params
df_parameters_N.dropna(inplace=True)
df_parameters_N.index = p[df_parameters_N.index].apply(lambda x: "$%s$" % str(x).replace('delta',r'\delta').replace('thrust','T'))
df_parameters_N.index.name = ''


# In[12]:


fig,axes=plt.subplots(ncols=3)
ax=axes[0]
df_parameters_X.plot.bar(ax=ax)
ax=axes[1]
df_parameters_Y.plot.bar(ax=ax)
ax.get_legend().set_visible(False)
ax=axes[2]
df_parameters_N.plot.bar(ax=ax)
plt.tight_layout()
ax.get_legend().set_visible(False)

