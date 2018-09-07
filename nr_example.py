import sys


from NelsonRulesClass import NelsonRules
del sys.modules['NelsonRulesClass']
from NelsonRulesClass import NelsonRules
from DescriptiveStatistics import DescriptiveStatistics

nr = NelsonRules()
nr.set_constant(3,7) #New constant for Rule 3 is set to 7 (previously 6)
nr.set_constant(6,9) #Constant for Rule 6 set to 9 (previously 4)
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
data_t = pd.read_csv('data.csv')
data = data_t

frames={}
figs={}

#ds = DescriptiveStatistics(outlier_thr = 3, dpi=200, format='pdf')
#DF = data.iloc[:,1:].copy()

for col in data.columns:
    frames[col],figs[col] = nr.apply_rules(data[col],var_name=col,rules=[1,3,5,7])
    figs[col].savefig('rules_'+col+'.png',format='png')
    frames[col].to_csv('frame_'+col+'.csv')
    #x  = DF[col].copy()
    print(col)
    #df_ds_temp = ds.plot(x,col,prefix='stat')
    #df_ds.append(df_ds_temp, ignore_index=True)
plt.close('all')
