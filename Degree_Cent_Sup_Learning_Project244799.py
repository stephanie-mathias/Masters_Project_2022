'''
Stephanie L. Mathias, August 2022. Masters Project: Network Parameters of Synesthete Connectomes.

This script runs a balanced random forest classifier on average degree centrality and for
the 360 node degree centralities for full and sparse datasets
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

# filepaths
full_data_file = r"_insert_fulldataset_filepath_"
sparse_data_file = r"_insert_sparsedataset_filepath_"

# import data file
full_data = pd.read_csv(full_data_file)
sparse_data = pd.read_csv(sparse_data_file)

# clean data
sparse_data['group'] = sparse_data['group'].str.strip()
sparse_data['group'] = np.where(sparse_data['group'] == 'control', 'control', 'syn')

# curate average degree centrality
data_params = full_data.columns.to_list()
nodes = data_params[:-1]
full_data_copy = full_data.copy()
sparse_data_copy = sparse_data.copy()
full_data_copy['av_degree'] = full_data_copy[nodes].mean(axis=1)
sparse_data_copy['av_degree'] = sparse_data_copy[nodes].mean(axis=1)

full_av_deg = full_data_copy[['av_degree','group']]
sparse_av_deg = sparse_data_copy[['av_degree','group']]

#Function: runs balanced random forest classifier with 5-fold cross validation, estimators = 500
def rand_forest_cv_500(dataset,cv,params):
    rounds = 0
    accs = []
    tpr_list = []
    tnr_list = []

    while rounds < cv:

        # state model
        classifier = BalancedRandomForestClassifier(n_estimators=500)

        # define x and y
        x = dataset[params]
        y = dataset['group']

        # split test and train
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # fit model
        classifier.fit(x_train, y_train)
        # make predictions
        y_pred = classifier.predict(x_test)
        y_pred_col = pd.Series(y_pred)

        # get accuracy
        acc = accuracy_score(y_test, y_pred_col)
        accs.append(acc)

        # get tpr and tnr
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        tpr_list.append(tpr)
        tnr_list.append(tnr)

        rounds += 1

    return [np.mean(accs),np.mean(tpr_list),np.mean(tnr_list)]

# run for average degree centrality full set
full_av_class = rand_forest_cv_500(full_av_deg,5,['av_degree'])

# run for average degree centrality sparse set
sparse_av_class = rand_forest_cv_500(sparse_av_deg,5,['av_degree'])

# run for 360 node degree centrality values full set
full_all_class = rand_forest_cv_500(full_data,5,nodes)

# run for 360 node degree centrality values sparse set
sparse_all_class = rand_forest_cv_500(sparse_data,5,nodes)

# plot bar chart

# prep data for bar chart
accuracy = [full_av_class[0],sparse_av_class[0],full_all_class[0],sparse_all_class[0]]
TPR = [full_av_class[1],sparse_av_class[1],full_all_class[1],sparse_all_class[1]]
TNR = [full_av_class[2],sparse_av_class[2],full_all_class[2],sparse_all_class[2]]
model = ['Av. Deg Full','Av. Deg Sparse','Deg Cent Full', 'Deg Cent Sparse']
perform_df = pd.DataFrame()
perform_df['model'] = model
perform_df['accuracy'] = accuracy
perform_df['TPR'] = TPR
perform_df['TNR'] = TNR
perform_melt = perform_df.melt(id_vars="model", var_name="metric")

# create bar chart
# Function: adds value labels to bars
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

sns.set(rc={'figure.figsize':(10,10)})
ax = sns.barplot(x="model", y="value", hue="metric",palette="Blues", data=perform_melt)
ax.set_title('Performance Metrics of Random Forest Classifier on Average Degree Centrality and Degree Centrality',fontsize=15)
sns.set(style="dark")
show_values(ax)
ax.set_xlabel("")
ax.set_ylabel("Performance",fontsize=15)
ax.set_xticklabels(['Av. Deg Cent (Full)','Av. Deg Cent (Sparse)','Deg Cent (Full)','Deg Cent (Sparse)'], fontsize = 15)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.95))

plt.tight_layout()
plt.show()
