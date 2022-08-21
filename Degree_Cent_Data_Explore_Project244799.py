'''
Stephanie L. Mathias, August 2022. Masters Project: Network Parameters of Synesthete Connectomes.

This script curates performs exploratory data analysis on the degree centrality datasets for full and sparse matrices.
Average degree centrality is calculated and boxplots are made for all data.
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# state filepaths
full_data = r"_insert_full_data_deg_cent_filepath_"
sparse_data = r"_insert_sparse_data_deg_cent_filepath_"

# import datafile
full_data = pd.read_csv(full_data)
sparse_data = pd.read_csv(sparse_data)

# get list of columns
data_params = full_data.columns.to_list()

# clean data
sparse_data['group'] = sparse_data['group'].str.strip()
sparse_data['group'] = np.where(sparse_data['group'] == 'control', 'control', 'syn')

# calculate average degree centrality
nodes = data_params[:-1]
full_data_copy = full_data.copy()
sparse_data_copy = sparse_data.copy()
full_data_copy['av_degree'] = full_data_copy[nodes].mean(axis=1)
sparse_data_copy['av_degree'] = sparse_data_copy[nodes].mean(axis=1)

# plot average degree centrality for two datasets
fig, axes = plt.subplots(1, 2, figsize=(9,10))
sns.set(style="dark")
sns.boxplot(y=full_data_copy['av_degree'], x=full_data_copy['group'], data=full_data_copy, orient='v',palette="Blues", ax=axes[0]).set(
    xlabel='',
    ylabel='Average Degree Centrality'
)
axes[0].set_title('Full Nodes Data')
sns.boxplot(y=sparse_data_copy['av_degree'], x=sparse_data_copy['group'], data=sparse_data_copy, orient='v',palette="Greens", ax=axes[1]).set(
    xlabel='',
    ylabel='Average Degree Centrality'
)
axes[1].set_title('Sparse Nodes Data')
fig.suptitle('Average Degree Centrality for Full and Sparse Average Degree Centrality',fontsize=15)

plt.tight_layout()
plt.show()

# plot boxplots for top 12 nodes for full dataset
fig, axes = plt.subplots(3, 4, figsize=(12,20))
plt.suptitle("Degree Centrality of 12 Nodes, Full Data Set",fontsize=15,y=0.99)
sns.boxplot(y=full_data_nc['R_V6'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[0][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][0].set_title('R_V6')
sns.boxplot(y=full_data_nc['R_7Pm'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[0][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][1].set_title('R_7Pm')
sns.boxplot(y=full_data_nc['R_V1'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[0][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][2].set_title('R_V1')
sns.boxplot(y=full_data_nc['R_V6'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[0][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][3].set_title('R_V6')
sns.boxplot(y=full_data_nc['R_V3'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[1][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][0].set_title('R_V3')
sns.boxplot(y=full_data_nc['R_PSL'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[1][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][1].set_title('R_PSL')
sns.boxplot(y=full_data_nc['R_6r'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[1][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][2].set_title('R_6r')
sns.boxplot(y=full_data_nc['R_55b'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[1][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][3].set_title('R_55b')
sns.boxplot(y=full_data_nc['R_V7'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[2][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][0].set_title('R_V7')
sns.boxplot(y=full_data_nc['R_23c'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[2][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][1].set_title('R_23c')
sns.boxplot(y=full_data_nc['R_a47r'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[2][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][2].set_title('R_a47r')
sns.boxplot(y=full_data_nc['R_9a'], x=full_data_nc['group'], data=full_data_nc,  orient='v',palette="Blues", ax=axes[2][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][3].set_title('R_9a')

sns.set(style="dark")

plt.tight_layout()
plt.show()

# plot boxplots for top 12 nodes for sparse dataset
my_pal = {"control": "g", "control":"b"}

fig, axes = plt.subplots(3, 4, figsize=(12,20))
plt.suptitle("Degree Centrality of 12 Nodes, Sparse Data Set",fontsize=15,y=0.99)
sns.boxplot(y=sparse_data['R_V6'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[0][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][0].set_title('R_V6')
sns.boxplot(y=sparse_data['R_V2'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[0][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][1].set_title('R_V2')
sns.boxplot(y=sparse_data['R_FEF'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[0][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][2].set_title('R_FEF')
sns.boxplot(y=sparse_data['R_V3'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[0][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[0][3].set_title('R_V3')
sns.boxplot(y=sparse_data['R_POS2'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[1][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][0].set_title('R_POS2')
sns.boxplot(y=sparse_data['R_7m'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[1][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][1].set_title('R_7m')
sns.boxplot(y=sparse_data['R_a9-46v'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[1][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][2].set_title('R_a9-46v')
sns.boxplot(y=sparse_data['R_TE1a'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[1][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[1][3].set_title('R_TE1a')
sns.boxplot(y=sparse_data['R_LO1'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[2][0]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][0].set_title('R_LO1')
sns.boxplot(y=sparse_data['R_MT'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[2][1]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][1].set_title('R_MT')
sns.boxplot(y=sparse_data['R_7AL'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[2][2]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][2].set_title('R_7AL')
sns.boxplot(y=sparse_data['R_BA45'], x=sparse_data['group'], data=sparse_data,  orient='v',palette="Greens", ax=axes[2][3]).set(
    xlabel='',
    ylabel='Degree Centrality'
)
axes[2][3].set_title('R_BA45')

sns.set(style="dark")

plt.tight_layout()
plt.show()

# get descriptive statistics for top 12 nodes
sparse_data_s = sparse_data[sparse_data['group']=='syn']
sparse_data_c = sparse_data[sparse_data['group']=='control']
full_data_s = full_data[full_data['group']=='syn']
full_data_c = full_data[full_data['group']=='control']
n12_cols = top_10_nodes
n12_cols.append('group')
s_data_s = sparse_data_s[n12_cols]
s_data_c = sparse_data_c[n12_cols]
f_data_s = full_data_s[n12_cols]
f_data_c = full_data_c[n12_cols]


s_data_s.describe().round(3)
s_data_c.describe().round(3)
f_data_s.describe().round(2)
f_data_c.describe().round(2)

# get descriptive statistics for average degree centrality
desc_col = ['av_degree','group']

full_cent = full_data_copy[desc_col]
sparse_cent = sparse_data_copy[desc_col]
s_full_cent = full_cent[full_cent['group']=='syn']
c_full_cent = full_cent[full_cent['group']=='control']
s_spar_cent = sparse_cent[sparse_cent['group']=='syn']
c_spar_cent = sparse_cent[sparse_cent['group']=='control']
s_full_cent.describe().round(2)
c_full_cent.describe().round(2)
s_spar_cent.describe().round(3)
c_spar_cent.describe().round(3)
