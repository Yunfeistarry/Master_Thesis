# coding: utf-8
import os
#import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
#import glob



# <h1>Training
# Load data
files_list=glob.glob('/cluster/home/wangyun/SimRes/*.txt')
c=np.arange(48,240)  # index of columns to be deleted

for i in files_list[0:1]:
    file = open("%s"%i, "r")
    sim_a=np.loadtxt(file)
    sim_A=np.delete(sim_a, c, axis=1)
    print sim_a.shape,sim_A.shape

for i in files_list[1:4231]:
    file = open("%s"%i, "r")
    sim_b=np.loadtxt(file)
    sim_B=np.delete(sim_b, c, axis=1)
    M=np.concatenate((sim_A,sim_B),axis=0)
    sim_A=M
print M.shape    #1-#4231 households' information matrix

SS_train=M[:,48] #Self-sufficiency
SC_train=M[:,49] #Self-Consumption

X_train=np.delete(M, [48,49,52,53], axis=1) #Feature Matrix: 48 load profile+ #PV + Battery Size
print X_train.shape
X_train=np.nan_to_num(X_train) #Returns an array or scalar replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative infinity with a very small (or negative) number.
SS_train=np.nan_to_num(SS_train)

# Preprocessing
from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train)
X_train_scaled.shape

# Train Model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
RF_regressor=RandomForestRegressor(n_estimators=20, criterion='mse',max_features=50, min_samples_split=2)
RF_regressor.fit(X_train_scaled,SS_train)
r2_train=RF_regressor.score(X_train_scaled,SS_train)
feat_imp=RF_regressor.feature_importances_

print('r_squared_value',r2_train,'feature_importance',feat_imp)

# Cross Validation
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=20) #for regressor, we use KFold, for classfiers, we use stratifiedKFold...
results = model_selection.cross_val_score(RF_regressor, X_train_scaled, SS_train, cv=kfold)
CV_result_mean=results.mean()
print('cv_results',results)
print('cv_results_mean',results.mean())








# <h1>Validating
# Load Data
files_list=glob.glob('./SimRes/*.txt')
c=np.arange(48,240)  # index of columns to be deleted

for i in files_list[3300:3301]:
    file = open("%s"%i, "r")
    sim_a=np.loadtxt(file)
    sim_A=np.delete(sim_a, c, axis=1)
    print sim_a.shape,sim_A.shape

for i in files_list[3301:3800]:
    file = open("%s"%i, "r")
    sim_b=np.loadtxt(file)
    sim_B=np.delete(sim_b, c, axis=1)
    M=np.concatenate((sim_A,sim_B),axis=0)
    sim_A=M
print M.shape    #501-#1000 households' feature matrix

SS_test=M[:,48]
SC_test=M[:,49]

X_test=np.delete(M, [48,49,52,53], axis=1)
print X_test.shape
X_test=np.nan_to_num(X_test) #Returns an array or scalar replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative infinity with a very small (or negative) number.
SS_test=np.nan_to_num(SS_test)


from sklearn import preprocessing
X_test_scaled = preprocessing.scale(X_test)
X_test_scaled.shape


# Predict with the trained model
SS_hat=RF_regressor.predict(X_test_scaled)
SS_hat.shape
from sklearn.metrics import r2_score
r2_test=r2_score(SS_test, SS_hat)
r2_test

# Check the difference between the prediction value and the simulation value
diff=SS_test-SS_hat
# plot(diff)
# plt.grid(True)
# plt.hist(diff,bins=80,color='c')
# plt.grid(True)
mean_diff=np.mean(diff)
print('mean_difference',mean_diff)



# Output
cv_output=np.array([r2_train,r2_test,CV_result_mean])
#output=np.concatenate((output,diff))
#output.size
#np.savetxt('output.txt',output, delimiter=',')
np.save('cv_output.out',cv_output)


# Save the Model
import pickle
from sklearn.externals import joblib
joblib.dump(RF_regressor, 'RandomForest.pkl') 
