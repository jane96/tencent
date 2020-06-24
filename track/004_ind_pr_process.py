import pandas as pd
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pr_data = pd.read_csv('/mnt/2TB/jane96/w2v/pr_ind/pr_click_all.csv').iloc[:,1:].astype(np.float16)
# ind_data.to_csv('/mnt/2TB/jane96/w2v/pr_ind/ind_click_all.csv',index=False)
# data = pd.read_csv('/mnt/2TB/jane96/w2v/pr_ind/pr_click2.csv').drop(['Unnamed: 0'],axis=1)
# data_sqrt = data.apply(lambda x : x.map(lambda y : math.sqrt(y)))
# data_square =  data.apply(lambda x :x.map(lambda y : y * y))
# result = pd.concat([data_sqrt,data,data_square],axis=1)
# result.to_csv('/mnt/2TB/jane96/w2v/pr_ind/pr_click_all.csv')

data = pd.read_csv('/mnt/2TB/jane96/w2v/pr_ind/ind_click2.csv').drop(['Unnamed: 0'],axis=1)
data_sqrt = data.apply(lambda x : x.map(lambda y : math.sqrt(y)))
data_square =  data.apply(lambda x :x.map(lambda y : y * y))
result = pd.concat([data_sqrt,data,data_square],axis=1)
result.to_csv('/mnt/2TB/jane96/w2v/pr_ind/ind_click_all.csv')
