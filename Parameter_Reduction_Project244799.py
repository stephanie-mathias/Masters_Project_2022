'''
Stephanie L. Mathias, July 2022. Masters Project: Network Parameters of Synesthete Connectomes.

This script runs logistic regression on the discovery dataset with varying class weights and L1 penalties.
The non-zero parameters from these models is used to create reduced datasets and these are run in a 5-fold
cross validation of a random forest classifier to obtain its performance on classifying synesthetes and controls.
This is run again on the analytical set for generalisation.
The best subset of parameters is exported to be used for curation of sparse connectome matrices.
'''
#import python packages
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

#list file paths of data files
discovery_set_file = r"_insert_discovery_set_filepath_"
parcel_IDs_set_file = r"_insert_parcelID_set_filepath_"
analytical_set_file = r"_insert_analytical_set_filepath_"

#import the files
disc_set = pd.read_csv(discovery_set_file)
parcel_IDS = pd.read_csv(parcel_IDs_set_file)
other_set = other_set = pd.read_csv(analytical_set_file)

#delete unused columns
del disc_set['Unnamed: 0.1']
del disc_set['Unnamed: 0']

#get colummn names in a list
disc_set_abs = disc_set.copy()
#get node numbers columns only
disc_set_cols = disc_set.columns.to_list()[:-2]

#get absolute values
for c in disc_set_cols:
    disc_set_abs[c] = np.absolute(disc_set_abs[c])
print(disc_set_abs)

#convert groups to 0 and 1
disc_set_abs['group_num'] = np.where(disc_set_abs['group'] == 'syn', 1, 0)

#curate two sets for logistic regression and random forest (50:50
disc_abs_s = disc_set_abs[disc_set_abs['group']=='syn']
disc_abs_c = disc_set_abs[disc_set_abs['group']=='control']
disc_abs_s1 = disc_abs_s.head(12)
disc_abs_s2 = disc_abs_s.tail(13)
disc_abs_c1 = disc_abs_c.head(88)
disc_abs_c2 = disc_abs_c.tail(87)
LR_disc = pd.concat([disc_abs_s1,disc_abs_c1])
RF_disc = pd.concat([disc_abs_s2,disc_abs_c2])

#split both sets into training and test sets (70:30)
x = LR_disc[disc_set_cols]
y = LR_disc['group_num']
x_r = RF_disc[disc_set_cols]
y_r = RF_disc['group_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(x_r, y_r, test_size=0.3, stratify=y)

#dictionaries to store non-zero parameters for each class weighting
non_zero_alpha_5 = {}
non_zero_alpha_25 = {}
non_zero_alpha_6 = {}
non_zero_alpha_26 = {}
non_zero_alpha_7 = {}

#range of L1 penalties tested in the logisitic regression model
penalty = [0.001,0.01,0.05,0.1,0.2,0.5,0.75,1,1.5,2,5,10,20,30,50,100]

#range of class weights tested in the logisitic regression model (control:synesthete)
class_weights = [1:10**5,2:10**6,1:10**6,2:10**6,1:10**7]

#lists to store performance scores against LR test set
lg_accs = []
lg_tpr = []
lg_tnr = []
#lists to store performance scores against LR training set
lg_train_accs = []
lg_train_tpr = []
lg_train_tnr = []

#Function: runs logistic regression and outputs performance of predictions of test and training set
def logreg_non0(x_t,y_t,x_te,y_te,a):

    # build model (class_weight parameter is altered each time)
    logreg = LogisticRegression(class_weight={0:1,1:10**6},random_state=0, C=a, penalty='l1', solver='liblinear').fit(x_t,y_t)

    # get model to predict for test and train
    y_pred = logreg.predict(x_te)
    y_pred_train = logreg.predict(x_t)

    # get accuracy for test and train
    acc = accuracy_score(y_te, y_pred)
    lg_accs.append(acc)

    acc_train = accuracy_score(y_t, y_pred_train)
    lg_train_accs.append(acc_train)

    # get tpr and tnr for test and train
    # test
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    lg_tpr.append(tpr)
    lg_tnr.append(tnr)
    # train
    tnt, fpt, fnt, tpt = confusion_matrix(y_t, y_pred_train).ravel()
    tprt = tpt/(tpt+fnt)
    tnrt = tnt/(tnt+fpt)
    lg_train_tpr.append(tprt)
    lg_train_tnr.append(tnrt)

    # get non-zero coefficients
    coeffs = logreg.coef_

    coeffs_list = coeffs.tolist()
    coeffs_list1 = coeffs_list[0]

    coeffs_dict = {}
    for i in range(len(disc_set_cols)):
        coeffs_dict[disc_set_cols[i]] = coeffs_list1[i]

    coeffs_dict_non0 = {x:y for x,y in coeffs_dict.items() if y!=0.0}

    params = list(coeffs_dict_non0.keys())

    # save non-zero coefficients
    non_zero_alpha_6[a] = params

#iterate over different L1 penalty values
for p in penalty:
    logreg_non0(x_train,y_train,x_test,y_test,p)

# get number of non-zero parameters
num_params = []
for e in non_zero_alpha_6.values():
    num_params.append(len(e))

#Function: runs logistic regression with no penalty
def logreg_non0_none(x_t,y_t,x_te,y_te):

    # build model
    logreg = LogisticRegression(class_weight={0:1,0:1,1:10**6},random_state=0, penalty='none').fit(x_t,y_t)

    # get model to predict for test and train
    y_pred = logreg.predict(x_te)
    y_pred_train = logreg.predict(x_t)

    # get accuracy for test and train
    acc = accuracy_score(y_te, y_pred)
    lg_accs.append(acc)

    acc_train = accuracy_score(y_t, y_pred_train)
    lg_train_accs.append(acc_train)

    # get tpr and tnr for test and train
    # test
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    lg_tpr.append(tpr)
    lg_tnr.append(tnr)
    # train
    tnt, fpt, fnt, tpt = confusion_matrix(y_t, y_pred_train).ravel()
    tprt = tpt/(tpt+fnt)
    tnrt = tnt/(tnt+fpt)
    lg_train_tpr.append(tprt)
    lg_train_tnr.append(tnrt)

    coeffs = logreg.coef_

    coeffs_list = coeffs.tolist()
    coeffs_list1 = coeffs_list[0]

    coeffs_dict = {}
    for i in range(len(disc_set_cols)):
        coeffs_dict[disc_set_cols[i]] = coeffs_list1[i]

    coeffs_dict_non0 = {x:y for x,y in coeffs_dict.items() if y!=0.0}
    print(len(coeffs_dict_non6))

    params = list(coeffs_dict_non6.keys())

    print(acc,tpr,tnr,acc_train,tprt,tnrt)

logreg_non0_none(x_train,y_train,x_test,y_test)

# plot number of parameters versus varying L1 penalty for five weight classes
plt.figure(figsize=(12,8))
plt.plot(np.log(penalty), params_1_105,marker='o',color='lightblue',label='weight 1:10^5')
plt.plot(np.log(penalty), params_1_205,marker='o',color='darkcyan',label='weight 1:20^5')
plt.plot(np.log(penalty), params_1_106,marker='o',color='steelblue',label='weight 1:10^6')
plt.plot(np.log(penalty), params_1_206,marker='o',color='royalblue',label='weight 1:20^6')
plt.plot(np.log(penalty), params_1_107,marker='o',color='lightgreen',label='weight 1:10^7')
plt.title('Changing L1 penalty and class weight of Logistic Regression - Impact on Number of Non-Zero Coefficients')
plt.xlabel('log(penalty)')
plt.legend(loc="upper left")
plt.grid(True)
plt.ylabel('Number of Non-Zero Parameters')
plt.show()

# plot performance of logistic regression (test set predictions), one plot per class weighting
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(15, 25))
fig.suptitle("Impact on Logistic Regression Performance: varying Class Weight and L1 Penalty, shown per Class Weight", fontsize=15,y=1.002)

# plot 1:10^5 performance
ax1.plot(params_1_105,acc_test_1_105,marker='o',color='skyblue',label='Accuracy')
ax1.plot(params_1_105,tpr_test_1_105,marker='o',color='limegreen',label='TPR')
ax1.plot(params_1_105,tnr_test_1_105,marker='o',color='teal',label='TNR')
ax1.grid(True)
ax1.title.set_text('Logistic Regression Performance on Test Set, Class Weight 1:10^5')
ax1.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax1.legend(loc="center right")

# plot 1:20^5 performance
ax2.plot(params_1_205,acc_test_1_205,marker='o',color='skyblue',label='Accuracy')
ax2.plot(params_1_205,tpr_test_1_205,marker='o',color='limegreen',label='TPR')
ax2.plot(params_1_205,tnr_test_1_205,marker='o',color='teal',label='TNR')
ax2.grid(True)
ax2.title.set_text('Logistic Regression Performance on Test Set, Class Weight 1:20^5')
ax2.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax2.legend(loc="center right")

# plot 1:10^6 performance
ax3.plot(params_1_106,acc_test_1_106,marker='o',color='skyblue',label='Accuracy')
ax3.plot(params_1_106,tpr_test_1_106,marker='o',color='limegreen',label='TPR')
ax3.plot(params_1_106,tnr_test_1_106,marker='o',color='teal',label='TNR')
ax3.grid(True)
ax3.title.set_text('Logistic Regression Performance on Test Set, Class Weight 1:10^6')
ax3.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax3.legend(loc="center right")


# plot 1:20^5 performance
ax4.plot(params_1_206,acc_test_1_206,marker='o',color='skyblue',label='Accuracy')
ax4.plot(params_1_206,tpr_test_1_206,marker='o',color='limegreen',label='TPR')
ax4.plot(params_1_206,tnr_test_1_206,marker='o',color='teal',label='TNR')
ax4.grid(True)
ax4.title.set_text('Logistic Regression Performance on Test Set, Class Weight 1:10^6')
ax4.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax4.legend(loc="center right")

# plot 1:10^7 performance
ax5.plot(params_1_107,acc_test_1_107,marker='o',color='skyblue',label='Accuracy')
ax5.plot(params_1_107,tpr_test_1_107,marker='o',color='limegreen',label='TPR')
ax5.plot(params_1_107,tnr_test_1_107,marker='o',color='teal',label='TNR')
ax5.grid(True)
ax5.title.set_text('Logistic Regression Performance on Test Set, Class Weight 1:10^6')
ax5.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax5.legend(loc="center right")
plt.tight_layout()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(15, 17))
fig.suptitle("Impact on Logistic Regression Performance: varying Class Weight and L1 Penalty, shown per Performance Score", fontsize=15,y=1.002)

# plot performance again, separate plot for each performance measure
# plot accuracy
ax1.plot(params_1_105,acc_test_1_105,marker='o',color='skyblue',label='10^5')
ax1.plot(params_1_205,acc_test_1_205,marker='o',color='darkblue',label='20^5')
ax1.plot(params_1_106,acc_test_1_106,marker='o',color='darkcyan',label='10^6')
ax1.plot(params_1_206,acc_test_1_206,marker='o',color='lightgreen',label='20^6')
ax1.plot(params_1_107,acc_test_1_107,marker='o',color='darkgreen',label='10^6')
ax1.title.set_text('Accuracy')

ax1.set(xlabel="Number of Non-Zero Coefficients",ylabel="Accuracy")
ax1.legend(loc="center right")

# plot tpr
ax2.plot(params_1_105,tpr_test_1_105,marker='o',color='skyblue',label='10^5')
ax2.plot(params_1_205,tpr_test_1_205,marker='o',color='darkblue',label='20^5')
ax2.plot(params_1_106,tpr_test_1_106,marker='o',color='darkcyan',label='10^6')
ax2.plot(params_1_206,tpr_test_1_206,marker='o',color='lightgreen',label='20^6')
ax2.plot(params_1_107,tpr_test_1_107,marker='o',color='darkgreen',label='10^6')
ax2.title.set_text('True Positive Rates')
ax2.set(xlabel="Number of Non-Zero Coefficients",ylabel="True Positive Rate")
ax2.legend(loc="center right")

# plot tnr
ax3.plot(params_1_105,tnr_test_1_105,marker='o',color='skyblue',label='10^5')
ax3.plot(params_1_205,tnr_test_1_205,marker='o',color='darkblue',label='20^5')
ax3.plot(params_1_106,tnr_test_1_106,marker='o',color='darkcyan',label='10^6')
ax3.plot(params_1_206,tnr_test_1_206,marker='o',color='lightgreen',label='20^6')
ax3.plot(params_1_107,tnr_test_1_107,marker='o',color='darkgreen',label='10^6')
ax3.title.set_text('True Negative Rates')
ax3.set(xlabel="Number of Non-Zero Coefficients",ylabel="True Negative Rate")
ax3.legend(loc="center right")

plt.tight_layout()
plt.show()

#Function: run balanced random forests with cross validation, estimators = 100
def rand_forest_cv(dataset,cv,params):
    rounds = 0
    accs = []
    tpr_list = []
    tnr_list = []

    while rounds < cv:

        # classifier model
        classifier = BalancedRandomForestClassifier(n_estimators=100)

        # set x and y parameters
        x = dataset[params]
        y = dataset['group']

        # split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # fit model
        classifier.fit(x_train, y_train)

        # get model to predict
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

    # return the means of performance scores
    return [np.mean(accs),np.mean(tpr_list),np.mean(tnr_list)]

ba_av_ac = []
ba_av_tpr = []
ba_av_tnr = []

# perform this for each subset of data coefficients
for key,params in non_zero_alpha_6.items():
    metrics = rand_forest_cv(RF_disc,5,params)
    ba_av_ac.append(metrics[0])
    ba_av_tpr.append(metrics[1])
    ba_av_tnr.append(metrics[2])

n_100_ac = [round(num, 2) for num in ba_av_ac]
n_100_tpr = [round(num, 2) for num in ba_av_tpr]
n_100_tnr = [round(num, 2) for num in ba_av_tnr]

#Function: run balanced random forests with cross validation, estimators = 500
def rand_forest_cv_500(dataset,cv,params):
    rounds = 0
    accs = []
    tpr_list = []
    tnr_list = []

    while rounds < cv:

        # classifier model
        classifier = BalancedRandomForestClassifier(n_estimators=500)

        # set x and y parameters
        x = dataset[params]
        y = dataset['group']

        # split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # fit model
        classifier.fit(x_train, y_train)

        # get model to predict
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

    # return the means of performance scores
    return [np.mean(accs),np.mean(tpr_list),np.mean(tnr_list)]

#store scores
ba_av_ac = []
ba_av_tpr = []
ba_av_tnr = []

#run for all subset of parameters
for key,params in non_zero_alpha_6.items():
    metrics = rand_forest_cv_500(RF_disc,5,params)
    ba_av_ac.append(metrics[0])
    ba_av_tpr.append(metrics[1])
    ba_av_tnr.append(metrics[2])

# round to 2 decimal places
n_500_ac = [round(num, 2) for num in ba_av_ac]
n_500_tpr = [round(num, 2) for num in ba_av_tpr]
n_500_tnr = [round(num, 2) for num in ba_av_tnr]

#Function: run balanced random forests with cross validation, estimators = 1000
def rand_forest_cv1000(dataset,cv,params):
    rounds = 0
    accs = []
    tpr_list = []
    tnr_list = []

    while rounds < cv:

        # classifier model
        classifier = BalancedRandomForestClassifier(n_estimators=1000)

        # set x and y parameters
        x = dataset[params]
        y = dataset['group']

        # split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # fit model
        classifier.fit(x_train, y_train)

        # get model to predict
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

    # return the means of performance scores
    return [np.mean(accs),np.mean(tpr_list),np.mean(tnr_list)]

# run for all sets of non-zero parameters
for key,params in non_zero_alpha_6.items():
    metrics = rand_forest_cv_1000(RF_disc,5,params)
    ba_av_ac.append(metrics[0])
    ba_av_tpr.append(metrics[1])
    ba_av_tnr.append(metrics[2])

n_1000_ac = [round(num, 2) for num in ba_av_ac]
n_1000_tpr = [round(num, 2) for num in ba_av_tpr]
n_1000_tnr = [round(num, 2) for num in ba_av_tnr]

# plot performance of the three random forest classifiers, one plot per number of classifiers
fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(15, 15))
fig.suptitle("Performance of Balanced Random Forest on Reduced-Size Datasets using Varied Numbers of Estimators", fontsize=15,y=1.002)

# plot estimators = 100 performance
ax1.plot(params_1_106,n_100_ac,marker='o',color='skyblue',label='Accuracy')
ax1.plot(params_1_106,n_100_tpr,marker='o',color='limegreen',label='TPR')
ax1.plot(params_1_106,n_100_tnr,marker='o',color='teal',label='TNR')
ax1.grid(True)
ax1.title.set_text('Random Forest Performance, Estimators = 100, Class Weight = 1:10^6')
ax1.set(xlabel="Number of Non-Zero Parameters in Dataset",ylabel="Performance Score")
ax1.legend(loc="upper right")

# plot estimators = 500 performance
ax2.plot(params_1_106,n_500_ac,marker='o',color='skyblue',label='Accuracy')
ax2.plot(params_1_106,n_500_tpr,marker='o',color='limegreen',label='TPR')
ax2.plot(params_1_106,n_500_tnr,marker='o',color='teal',label='TNR')
ax2.grid(True)
ax2.title.set_text('Random Forest Performance, Estimators = 500, Class Weight = 1:10^6')
ax2.set(xlabel="Number of Non-Zero Parameters in Dataset",ylabel="Performance Score")
ax2.legend(loc="upper right")

# plot estimators = 1000 performance
ax3.plot(params_1_106,n_1000_ac,marker='o',color='skyblue',label='Accuracy')
ax3.plot(params_1_106,n_1000_tpr,marker='o',color='limegreen',label='TPR')
ax3.plot(params_1_106,n_1000_tnr,marker='o',color='teal',label='TNR')
ax3.grid(True)
ax3.title.set_text('Random Forest Performance, Estimators = 1000, Class Weight = 1:10^6')
ax3.set(xlabel="Number of Non-Zero Parameters in Dataset",ylabel="Performance Score")
ax3.legend(loc="upper right")

plt.tight_layout()
plt.show()

# clean up analytical set
other_set['group'] = other_set.group.str.strip()
other_set.group.unique()
other_set_copy = other_set.copy()
del other_set_copy['ID']
del other_set_copy['Unnamed: 0']
other_set_copy['group'] = np.where(other_set_copy['group'] == 'control', 'control', 'syn')
print(other_set_copy)

# convert to absolute values
prs = [str(x) for x in range(1,129601)]
for pr in prs:
    other_set_copy[pr] = np.abs(other_set_copy[pr])
print(other_set_copy)

# run random forest (estimators = 500) on analytical set
for key,params in non_zero_alpha_6.items():
    metrics = rand_forest_cv_500(other_set_copy,5,params)
    ba_av_ac_other.append(metrics[0])
    ba_av_tpr_other.append(metrics[1])
    ba_av_tnr_other.append(metrics[2])

fig, ax = plt.subplots(1,figsize=(10, 5))

# plot results on analytical set
ax.plot(params_1_106,round_ba_av_ac_other,marker='o',color='skyblue',label='Accuracy')
ax.plot(params_1_106,round_ba_av_tpr_other,marker='o',color='limegreen',label='TPR')
ax.plot(params_1_106,round_ba_av_tnr_other,marker='o',color='teal',label='TNR')
ax.grid(True)
ax.title.set_text('Random Forest Performance on Analysis Set, Estimators = 500, Class Weight = 1:10^6')
ax.set(xlabel="Number of Non-Zero Coefficients",ylabel="Performance Score")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

# convert best parameters (1893) to dataframe
params_1893 = pd.DataFrame()
params_1893['params'] = non_zero_alpha_6[0.1]
print(params_1893)
