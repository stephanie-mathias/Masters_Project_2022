'''
Stephanie L. Mathias, August 2022. Masters Project: Network Parameters of Synesthete Connectomes.

This script curates degree centrality for sparse and full datasets.
The sparse datasets come from reduced parameters from logistic regression.
The datasets per subject with degree centrality are exported.
'''
# import libraries
import pandas as pd
import numpy as np
from numpy import arange
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# file paths
reduced_parameter_file = r"_insert_reducedparam_filepath_"
full_dataset_file = r"_insert_fulldataset_filepath_"
analytical_set_file = r"_insert_analytical_set_filepath_"
parcel_IDs_set_file = r"_insert_parcelID_set_filepath_"
subject_id_file = r"_insert_subjectid_filepath_"
other_id_file = r"_insert_other_ids_file_"
discovery_file = r"_insert_discovery_set_filepath_"
parcel_names_file = r"_insert_parcel_names_filepath_"

# import files
reduced_params = pd.read_csv(reduced_parameter_file)
parcel_IDS = pd.read_csv(parcel_IDs_set_file)
full_set = pd.read_csv(full_dataset_file)
subject_ID = pd.read_csv(subject_id_file)
other_set = pd.read_csv(analytical_set_file)
other_ID_group = pd.read_csv(other_id_file)
disc_set = pd.read_csv(discovery_file)
parcel_names = pd.read_csv(parcel_names_file)

# list of node parameters
ps = [x for x in range(1,129601)]

# clean data
full_set["ID"] = subject_ID["ID"]

# map ids and classes onto datasets
other_list = other_ID_group.ID.to_list()
other_group = other_ID_group.group.to_list()
disc_list = disc_set.ID.to_list()
disc_group = disc_set.group.to_list()

IDs = []
group = []
for i in other_list:
    IDs.append(i)
for j in other_group:
    group.append(j)
for k in disc_list:
    IDs.append(k)
for l in disc_group:
    group.append(l)

IDs_dict = {}
indexes = [x for x in range(0,444)]
for i in indexes:
    ID = IDs[i]
    grp = group[i]
    IDs_dict[ID] = grp

full_set['group'] = full_set['ID'].map(IDs_dict)

# clean data
full_set['group'] = full_set.group.str.strip()
full_set['group'] = np.where(full_set['group'] == 'control', 'control', 'syn')

# convert to absolute values
prs = [x for x in range(1,129601)]
for pr in prs:
    full_set[pr] = np.abs(full_set[pr])
    other_set[str(pr)] = np.abs(other_set[str(pr)])

# get reduced dataset parameters and curate analytical set with these
red_params_copy = reduced_params['params'].to_list()
reduced_params['params'] = reduced_params['params'].astype(str)
red_params = reduced_params['params'].to_list()
red_other_set = other_set[red_params]

# convert full dataset to 360 x 360 matrices
full_groups = []
full_matrices = []

#Function: takes list and converts to matrix
def corr_matrix(data):

    data_a = np.array(data)
    shape = (360,360)
    mat_corr = data_a.reshape(shape)

    return mat_corr

# convert all subjects in full set to matrices
for index, row in full_set.iterrows():
    grp = row['group']
    mat_rows = row.to_list()
    mat_rows = mat_rows[:-2]

    mat_c = corr_matrix(mat_rows)

    full_groups.append(grp)
    full_matrices.append(mat_c)

# get degree centrality for all 360 nodes of full data matrices
deg_360 = []

for m in full_matrices:
    m_sum = m.sum(axis=1, dtype='float')
    norm_m = []
    for n in m_sum:
        nn = n / 360
        norm_m.append(nn)

    deg_360.append(norm_m)

# import node names
region_names = parcel_names['parcel_name'].to_list()
# apply as column names to degree centrality dataframe
full_degree_cent = pd.DataFrame(deg_360,columns=region_names)
full_degree_cent['group'] = full_groups
#export dataframe
full_degree_cent.to_csv(r"_export_filepath_")

# clean analytical set
del other_set['Unnamed: 0']
other_set_copy = other_set.copy()
red_ps_cm = []

# get column node names that should be zero
for r in red_params_copy:
    red_ps_cm.append(str(r))

colls = [str(x) for x in range(1,129601)]
keep_cols = []
not_cols = []
for c in colls:
    if c in red_ps_cm:
        keep_cols.append(c)
    else:
        not_cols.append(c)

# convert columns to zero
for col in not_cols:
    other_set_copy[col].values[:] = 0

# create sparse full_matrices
sparse_groups = []
sparse_matrices = []

for index, row in other_set_copy.iterrows():
    grp = row['group']
    mat_rows = row.to_list()
    mat_rows = mat_rows[:-2]

    mat_c = corr_matrix(mat_rows)

    sparse_groups.append(grp)
    sparse_matrices.append(mat_c)

# count non-zero connections per node
non_zeros_nodes = []
for i in sparse_matrices[0]:
    count = 0
    for j in i:
        if j != 0:
            count += 1
    non_zeros_nodes.append(count)

non_zeros = pd.DataFrame()
non_zeros['non_zero_nodes'] = non_zeros_nodes
print(non_zeros)
# export non-zero connections count per node
non_zeros.to_excel(r"_insert_export_filepath_")

# calculate sparse matrices degree centrality
deg_360_sparse = []

for m in sparse_matrices:
    m_sum = m.sum(axis=1, dtype='float')
    norm_m = []
    for n in m_sum:
        nn = n / 360
        norm_m.append(nn)

    deg_360_sparse.append(norm_m)

# create dataframe for sparse degree centrality values
sparse_degree_cent = pd.DataFrame(deg_360_sparse,columns=region_names)
sparse_degree_cent['group'] = sparse_groups
print(sparse_degree_cent)

# export sparse degree centrality dataframe
sparse_degree_cent.to_csv(r"_insert_export_filepath_")
