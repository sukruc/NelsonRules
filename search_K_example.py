import sys


from NelsonRulesClass import NelsonRules
del sys.modules['NelsonRulesClass']
from NelsonRulesClass import NelsonRules
#from DescriptiveStatistics import DescriptiveStatistics

nr = NelsonRules()
nr.set_constant(3,7) #New constant for Rule 3 is set to 7 (previously 6)
nr.set_constant(6,9) #Constant for Rule 6 set to 9 (previously 4)
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')

info_lost={}
df={}


for col in data.columns:
    df[col],info_lost[col]=nr.search_K(data[col],var_name=col,rule=2,K_list=range(8,22),plots=False) # Searches for K within range 8-22 for Rule 2, doesn't output plots
    print(col)
plt.close('all')
