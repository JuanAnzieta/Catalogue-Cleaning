#!/usr/bin/env python3
#########INTEL OPTIMIZER COMMENT IF NOT INTEL CPU
from sklearnex import patch_sklearn
patch_sklearn()
#################GENERAL MODULES
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
##################DISABLE WARNINGS
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
###################MODELS######################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



volcano_name='Cotopaxi'############MODIFY VOLCANO/CATALOG NAME HERE AS REQUIRED

features_group='watson'######MODIFY FEATURES GROUP NAME HERE FOR EACH FEATURE GROUP [watson, titos, sotoaj]

version='v0'#############MODIFY THIS ACCORDING TO THE CLEANING ITERATION

Originals=pd.read_csv("./Catalogue/"+volcano_name+"_Catalogue_"+version+".csv",header=None)

original_labels=np.array(Originals.iloc[:,1])

dictionary={'VT': 1, 'LP': 0}#RENAME TO OTHER EVENTS OF INTEREST IF NEEDED
dictionary2={0: 'LP', 1: 'VT'}#RENAME TO OTHER EVENTS OF INTEREST IF NEEDED


eventdata=pd.read_csv("./Features/"+volcano_name+"_features_"+features_group+".csv",header=None)

################PREDICTIONS

modelnames=['KNN','LOGiSTIC','lSVMsd','pSVMsd','rSVMsd','sNN','RF']
newlabels=np.zeros((len(Originals)+1,len(modelnames)+1),dtype=np.dtype('U17'))
newlabels[0,1:]=modelnames

for i in range(np.shape(newlabels)[0]-1):
	train_set=eventdata.drop(i)
	train_labels=Originals.drop(i)
	X_train=np.array(train_set.iloc[:,1:])
	Y_train=np.ravel(train_labels.iloc[:,1:])
	Y_train=np.vectorize(dictionary.get)(Y_train)

	X_predicted=np.array(eventdata.iloc[i,1:]).reshape(1, -1)
	
	
	####KNN
	knn_clf=KNeighborsClassifier(n_neighbors=1)##################################################
	knn_clf.fit(X_train, Y_train)
	newlabels[i+1][1]=dictionary2.get(knn_clf.predict(X_predicted)[0])
	
	####LOGISTIC
	logis_clf = LogisticRegression(solver="lbfgs", random_state=42, max_iter=5000,C=1)##########
	logis_clf.fit(X_train, Y_train)
	newlabels[i+1][2]=dictionary2.get(logis_clf.predict(X_predicted)[0])
	
	###SVM-linear-sd
	SVMl_clf_sd = Pipeline([
		("scaler",StandardScaler()),
		("svm_clf",SVC(kernel="linear",C=1))])#################################################
	SVMl_clf_sd.fit(X_train, Y_train)
	newlabels[i+1][3]=dictionary2.get(SVMl_clf_sd.predict(X_predicted)[0])
	
	###SVM-poly-sd
	SVMp_clf_sd = Pipeline([
		("scaler",StandardScaler()),
		("svm_clf",SVC(kernel="poly",degree=3,coef0=1,C=1))])##################################
	SVMp_clf_sd.fit(X_train, Y_train)
	newlabels[i+1][4]=dictionary2.get(SVMp_clf_sd.predict(X_predicted)[0])
	
	###SVM-RBF-sd
	SVMrbfsd_clf = Pipeline([
			("scaler",StandardScaler()),
			("svm_clf",SVC(kernel="rbf",gamma='scale',C=1))])########################################
	SVMrbfsd_clf.fit(X_train, Y_train)
	newlabels[i+1][5]=dictionary2.get(SVMrbfsd_clf.predict(X_predicted)[0])
	
	###sNN
	snnsd_clf = Pipeline([
			("scaler",StandardScaler()),
			("snn_clf",MLPClassifier(alpha=0.0001,hidden_layer_sizes=(100,),max_iter=1000,random_state=42,activation='relu',warm_start=True))])
	snnsd_clf.fit(X_train, Y_train)
	newlabels[i+1][6]=dictionary2.get(snnsd_clf.predict(X_predicted)[0])
	
	###RF
	RF_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_features='sqrt', random_state=42,n_jobs=7)
	RF_clf.fit(X_train, Y_train)
	newlabels[i+1][7]=dictionary2.get(RF_clf.predict(X_predicted)[0])

	print('progress: '+str(round(100*(i+1)/(np.shape(newlabels)[0]-1),2))+'%')



newlabels[1:,0]=np.array(Originals.iloc[:,0])

DF=pd.DataFrame(newlabels)
DF.to_csv('./Predictions/'+volcano_name+'_predictions_'+features_group+'_'+version+'.csv',index=False,header=False)