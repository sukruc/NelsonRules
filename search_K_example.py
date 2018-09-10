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

#looping through whole data frame:
for col in data.columns:
    df[col],info_lost[col]=nr.search_K(data[col],
                                        var_name=col,
                                        rule=9,
                                        K_list=range(8,22),
                                        plots=True,
                                        prefix='sK',
                                        dpi=200) # Searches for K within range 8-22 for Rule 9, outputs plots
    print(col)
plt.close('all')

#to retrieve the percentage of data loss for a given K (i.e. 12), use:
#>>>info_lost[col]['K=12']

#to retrieve results of different constants for a column, use:
#>>>df[col]

#or to retrieve results of a specific K value (i.e. 12), use:
#>>>df[col]['K=12']
