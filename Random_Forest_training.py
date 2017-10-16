# coding: utf-8
import os
#import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import glob



# <h1>Training
# Load data
files_list=glob.glob('/cluster/home/wangyun/SimRes/*.txt')
c=np.arange(48,240)  # index of columns to be deleted

for i in files_list[0:1]:
    file = open("%s"%i, "r")
    sim_a=np.loadtxt(file)
    sim_A=np.delete(sim_a, c, axis=1)
    print sim_a.shape,sim_A.shape

for i in files_list[1:4232]:
    file = open("%s"%i, "r")
    sim_b=np.loadtxt(file)
    sim_B=np.delete(sim_b, c, axis=1)
    M=np.concatenate((sim_A,sim_B),axis=0)
    sim_A=M
print M.shape    #1-#4231 households' information matrix

SS_train=M[:,48] #Self-sufficiency
SC_train=M[:,49] #Self-Consumption

X_raw=np.delete(M, [48,49,52,53], axis=1) #Feature Matrix: 48 load profile+ #PV + Battery Size
print X_raw.shape
X_raw=np.nan_to_num(X_raw) #Returns an array or scalar replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative infinity with a very small (or negative) number.
SS_train=np.nan_to_num(SS_train)

# Preprocessing
X2=X_raw[:,0:48]*0.5*365
X_50_unscaled=np.concatenate((X2,X_raw[:,[48,49]]),1)
print('feature matrix shape',X_50_unscaled.shape)



#split the data
from sklearn.model_selection import train_test_split #split the dataset, 90% for training and validation, 10% for testing
X_training, X_test, Y_training, Y_test = train_test_split(X_50_unscaled,SS_train, test_size=0.1, random_state=42)


# Train Model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
RF_regressor=RandomForestRegressor(n_estimators=20, criterion='mse',max_features=50, min_samples_split=2)
RF_regressor.fit(X_training,Y_training)
r2_train=RF_regressor.score(X_training,Y_training)
feat_imp=RF_regressor.feature_importances_

print('r_squared_value',r2_train,'feature_importance',feat_imp)

# Cross Validation
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=20) #for regressor, we use KFold, for classfiers, we use stratifiedKFold...
results = model_selection.cross_val_score(RF_regressor, X_training, Y_training, cv=kfold)
CV_result_mean=results.mean()
print('cv_results',results)
print('cv_results_mean',results.mean())





# Predict with the trained model
SS_hat=RF_regressor.predict(X_test)
SS_hat.shape
from sklearn.metrics import r2_score
r2_test=r2_score(Y_test, SS_hat)
r2_test

# Check the difference between the prediction value and the simulation value
diff=Y_test-SS_hat
# plot(diff)
# plt.grid(True)
# plt.hist(diff,bins=80,color='c')
# plt.grid(True)
mean_diff=np.mean(diff)
var_diff=np.var(diff)
print('mean_difference',mean_diff,'variance difference',var_diff)



# Output
cv_stats=np.array([r2_train,r2_test,CV_result_mean])
cv_scores=np.array(results)
feature_weight=np.array(feat_imp)
#output=np.concatenate((output,diff))
#output.size
#np.savetxt('output.txt',output, delimiter=',')
np.save('cv_stats.out',cv_stats)
np.save('cv_scores.out',cv_scores)
np.save('feature_weights.out',feat_imp)


# Save the Model
import pickle
from sklearn.externals import joblib
joblib.dump(RF_regressor, 'RandomForest.pkl') 
