import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''https://github.com/tannerdietrich/nelsonRules'''

class NelsonRules:

    def __init__(self):
        self.rule_dict={1:3,2:9,3:6,4:14,5:2,6:4,7:15,8:8}
        self.rule_expl={'1':' - Outlier('+str(self.rule_dict[1])+'$\sigma$)',
                        '2':' - Prolonged bias',
                        '3':' - Trend exists',
                        '4':' - Heavy oscillation',
                        '5':' - Mediumly out of control',
                        '6':' - Consequent samples on the same side',
                        '7':' - Very small variation',
                        '8':' - Sudden and high deviation'}
        self.__glob_rules_= ['rule1','rule2','rule3','rule4','rule5','rule6','rule7','rule8']
        # TODO: Pass rule numbers to initialize NelsonRules instance
        # TODO: Add a new attribute: self.rules
        return

    def set_constant(self,rule_num,constant):
        '''Updates rule constants dict with given value
        Example:
        >>>nr = NelsonRules()
        >>>print(nr.rule_dict)
        {1:3,2:9,3:6,4:14,5:2,6:4,7:15,8:8}
        >>>nr.set_constant(3,8) #New constant for rule 3 is 8
        >>>print(nr.rule_dict)
        {1:3,2:9,3:8,4:14,5:2,6:4,7:15,8:8}
        '''
        self.rule_dict[rule_num]=constant


    def _sliding_chunker(self,original, segment_len, slide_len):
        """Split a list into a series of sub-lists...
        each sub-list window_len long,
        sliding along by slide_len each time. If the list doesn't have enough
        elements for the final sub-list to be window_len long, the remaining data
        will be dropped.
        e.g. sliding_chunker(range(6), window_len=3, slide_len=2)
        gives [ [0, 1, 2], [2, 3, 4] ]
        """
        chunks = []
        for pos in range(0, len(original), slide_len):
            chunk = np.copy(original[pos:pos + segment_len])
            if len(chunk) != segment_len:
                continue
            chunks.append(chunk)
        return chunks


    def _clean_chunks(self,original, modified, segment_len):
        """appends the output argument to fill in the gaps from incomplete chunks"""
        results = []
        results = modified
        for i in range(len(original) - len(modified)):
            results.append(False)

        # set every value in a qualified chunk to True
        for i in reversed(range(len(results))):
            if results[i] == True:
                for d in range(segment_len):
                    results[i+d] = True

        return results


    def control_chart(self,original):
        """Plot control chart"""
        text_offset = 70
        mean = original.mean()
        sigma = original.std()

        # plot original
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(original, color='blue', linewidth=1.5)

        # plot mean
        ax1.axhline(mean, color='r', linestyle='--', alpha=0.5)
        ax1.annotate('$\overline{x}$', xy=(original.index.max(), mean), textcoords=('offset points'),
                     xytext=(text_offset, 0), fontsize=18)

        # plot 1-3 standard deviations
        sigma_range = np.arange(1,4)
        for i in range(len(sigma_range)):
            ax1.axhline(mean + (sigma_range[i] * sigma), color='black', linestyle='-', alpha=(i+1)/10)
            ax1.axhline(mean - (sigma_range[i] * sigma), color='black', linestyle='-', alpha=(i+1)/10)
            ax1.annotate('%s $\sigma$' % sigma_range[i], xy=(original.index.max(), mean + (sigma_range[i] * sigma)),
                         textcoords=('offset points'),
                         xytext=(text_offset, 0), fontsize=18)
            ax1.annotate('-%s $\sigma$' % sigma_range[i],
                         xy=(original.index.max(), mean - (sigma_range[i] * sigma)),
                         textcoords=('offset points'),
                         xytext=(text_offset, 0), fontsize=18)

        return fig

        # TODO: remove limit lines except for rule1
    def plot_rules(self,data,chart_type=1,var_name='variable',prefix='rules_'):
        if chart_type == 1:
            columns = data.columns[1:]
            fig, axs = plt.subplots(len(columns), 1, figsize=(20, 20),sharex=True, sharey=False)
            fig.subplots_adjust(hspace=.5, wspace=.5)
            plt.suptitle(var_name)
            legends={}

            #axs = axs.ravel()

            for i in range(len(columns)):
               axs[i].plot(data.ix[:, 0],label='Data')
               axs[i].plot(data.ix[:, 0][(data.ix[:, i+1] == True)], 'ro',label='Violations')
               #axs[i].set_title(columns[i]+' K='+str(self.rule_dict[i+1])) uncomment to activate K in title per rule
               print(columns[i])
               axs[i].set_title(columns[i]+' '+self.rule_expl[str(self.__glob_rules_.index(columns[i])+1)])
               mean = data.ix[:,0].mean()
               std = data.ix[:,0].std()
               axs[i].axhline(mean,color='g',label=r'$\mu$', linewidth=.3)
               if columns[i] == 'rule5':
                   axs[i].axhline(mean+std*2,color='b',linestyle='--',label=r'$2\sigma$', linewidth=.5)
                   axs[i].axhline(mean+std*-2,color='b',linestyle='--', linewidth=.5)
               if columns[i] == 'rule1':
                    axs[i].axhline(mean+std*3,color='r',linestyle='--',label=r'$3\sigma$', linewidth=.7)
                    axs[i].axhline(mean+std*-3,color='r',linestyle='--', linewidth=.7)
               if columns[i] in ['rule6','rule7','rule8']:
                    axs[i].axhline(mean+std,color='k',linestyle='--',label=r'$\sigma$', linewidth=.5)
                    axs[i].axhline(mean-std,color='k',linestyle='--', linewidth=.5)
            for i in range(len(columns)):
                if i != range(len(columns))[-1]:
                    axs[i].legend(loc='upper center', bbox_to_anchor=(.5,-.05),fancybox=True,ncol=5)
                else:
                    axs[i].legend(loc='upper center', bbox_to_anchor=(.5,-.15),fancybox=True,ncol=5)
            fig.savefig(prefix+'_'+var_name+'.png',format='png')
            plt.close()
            return

        elif chart_type == 2:
            # plot_num = len(data.columns[1:])
            fig = plt.figure(figsize=(20, 10))
            axs = fig.add_subplot(111)
            axs.plot(data.ix[:, 0])
            plt.title('Nelson Rules for '+var_name)
            marker = ['H', '+', '.', 'o', '*', '<', '>', '^']
            columns = data.columns[1:]

            for i in range(len(data.columns[1:])):
                axs.plot(data.ix[:, 0][(data.ix[:, i+1] == True)], ls='', marker=marker[i], markersize=20, label=columns[i])

            plt.legend()
            fig.savefig(prefix+'_'+var_name+'.png',format='png')
            plt.close()

            return


    def apply_rules(self,original=None, rules='all', chart_type=1,var_name='',prefix=''):
        '''Applies selected rules(default=all) to a given Pandas series object
        Returns a DataFrame with labels for each data point for given rules.
        True indicates violation
        Example:
        >>>rule_table,fig = nelsonRules.apply_rules(df['col'],var_name='col',prefix='save_filename_')
        see NelsonRules.set_constant() for changing rule constants.
        '''
        assert(type(original)!=pd.DataFrame),'original must be a pandas series object'
        if (original.dtype=='O'): # object
            print('----> Error: variable [%s] is Object' % var_name)
            return(pd.DataFrame(),plt.figure())
        if not original.ndim==1: # dim
            print('----> Error: Dim [%s] is not 1' % var_name)
            return(pd.DataFrame(),plt.figure())

        mean = original.mean()
        sigma = original.std()
        rule_dict = self.rule_dict
        rule_handle = [self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7, self.rule8]
        rule_nums = rules
        if rules == 'all':
            rules = rule_handle
        elif isinstance(rules,list):
            rules = [rule_handle[i-1] for i in rule_nums]
        df = pd.DataFrame(original)
        for i in range(len(rules)):
            df[rules[i].__name__] = rules[i](original, mean, sigma, K=rule_dict[rule_handle.index(rules[i])+1])


        self.plot_rules(df, chart_type,var_name=var_name,prefix=prefix)

        return df


    def rule1(self,original, mean=None, sigma=None,K=3):
        """One point is more than 3 standard deviations from the mean."""
        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        copy_original = original
        ulim = mean + (sigma * K)
        llim = mean - (sigma * K)

        results = []
        for i in range(len(copy_original)):
            if copy_original[i] < llim:
                results.append(True)
            elif copy_original[i] > ulim:
                results.append(True)
            else:
                results.append(False)

        return results


    def rule2(self,original, mean=None, sigma=None, K=9):
        """Nine (or more) points in a row are on the same side of the mean."""
        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        copy_original = original
        segment_len = K

        side_of_mean = []
        for i in range(len(copy_original)):
            if copy_original[i] > mean:
                side_of_mean.append(1)
            else:
                side_of_mean.append(-1)

        chunks =self._sliding_chunker(side_of_mean, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            if chunks[i].sum() == segment_len or chunks[i].sum() == (-1 * segment_len):
                results.append(True)
            else:
                results.append(False)

        # clean up results
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule3(self,original, mean=None, sigma=None, K=6):
        """Six (or more) points in a row are continually increasing (or decreasing)."""
        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            chunk = []

            # Test the direction with the first two data points and then iterate from there.
            if chunks[i][0] < chunks[i][1]: # Increasing direction
                for d in range(len(chunks[i])-1):
                    if chunks[i][d] < chunks[i][d+1]:
                        chunk.append(1)
            else: # decreasing direction
                for d in range(len(chunks[i])-1):
                    if chunks[i][d] > chunks[i][d+1]:
                        chunk.append(1)

            if sum(chunk) == segment_len-1:
                results.append(True)
            else:
                results.append(False)

        # clean up results
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule4(self,original, mean=None, sigma=None, K=14):
        """Fourteen (or more) points in a row alternate in direction, increasing then decreasing."""
        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            current_state = 0
            for d in range(len(chunks[i])-1):
                # direction = int()
                if chunks[i][d] < chunks[i][d+1]:
                    direction = -1
                else:
                    direction = 1

                if current_state != direction:
                    current_state = direction
                    result = True
                else:
                    result = False
                    break

            results.append(result)

        # fill incomplete chunks with False
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule5(self,original, mean=None, sigma=None, K=2):
        """Two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same
        direction."""

        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            if all(i > (mean + sigma * 2) for i in chunks[i]) or all(i < (mean - sigma * 2) for i in chunks[i]):
                results.append(True)
            else:
                results.append(False)

        # fill incomplete chunks with False
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule6(self,original, mean=None, sigma=None, K=4):
        """Four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same
        direction."""

        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            if all(i > (mean + sigma) for i in chunks[i]) or all(i < (mean - sigma) for i in chunks[i]):
                results.append(True)
            else:
                results.append(False)

        # fill incomplete chunks with False
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule7(self,original, mean=None, sigma=None, K=15):
        """Fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean."""

        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            if all((mean - sigma) < i < (mean + sigma) for i in chunks[i]) :
                results.append(True)
            else:
                results.append(False)

        # fill incomplete chunks with False
        results = self._clean_chunks(copy_original, results, segment_len)

        return results


    def rule8(self,original, mean=None, sigma=None, K=8):
        """Eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both
        directions from the mean."""

        if mean is None:
            mean = original.mean()

        if sigma is None:
            sigma = original.std()

        segment_len = K
        copy_original = original
        chunks = self._sliding_chunker(copy_original, segment_len, 1)

        results = []
        for i in range(len(chunks)):
            if all(i < (mean - sigma) or i > (mean + sigma) for i in chunks[i])\
                    and any(i < (mean - sigma) for i in chunks[i])\
                    and any(i > (mean + sigma) for i in chunks[i]):
                results.append(True)
            else:
                results.append(False)

        # fill incomplete chunks with False
        results = self._clean_chunks(copy_original, results, segment_len)

        return results

    def main(self,original, prefix='',img_format='png'):
        """Accepts DataFrame as input and returns for every column in dataframe an image file of specified format,
        which contains 8 different plots belonging to implementation of 8 different rules, and a *.csv file
        containing True/False for every data point. For a given data point and rule, True represents violation of the respective
        rule and False indicates no violation.
        Example:
        >>>nr = NelsonRules()
        >>>import pandas as pd
        >>>import matplotlib.pyplot as plt
        >>>data_t = pd.read_csv('data.csv')
        >>>data = data_t.iloc[:,1:]
        >>>data.shape
        (1000,8)
        >>>nr.main(data)
        <creates 8 image files and 8 csv files>"""
        figs = {}
        frames = {}
        copy_original = original
        for i in copy_original.columns:
            frames[i],figs[i] = self.apply_rules(original=copy_original[i])
            figs[i].savefig(prefix+'rules_'+i+'.png',format='png')
            frames[i].to_csv(prefix+'frame_'+i+'.csv')
        plt.close('all')
