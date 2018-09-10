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


for col in data.columns:
    nr.apply_rules(data[col],var_name=col,rules=[1,3,5,9])
    print(col)


plt.close('all')
