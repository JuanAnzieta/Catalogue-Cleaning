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

features_group='watson'######MODIFY FEATURES GROUP NAME HERE FOR EACH FEATURE GROUP

version='v0'#############MODIFY THIS ACCORDING TO THE CLEANING ITERATION


Originals=pd.read_csv("./Catalogue/"+volcano_name+"_catalogue_"+version+".csv",header=None)

original_labels=np.array(Originals.iloc[:,1])

dictionary={'VT': 1, 'LP': 0}#RENAME TO OTHER EVENTS OF INTEREST IF NEEDED


eventdata=pd.read_csv("./Features/"+volcano_name+"_features_"+features_group+".csv",header=None)

###################GRAND CV######################

V=10


n2 = len(eventdata)
np.random.seed(42)
folds = np.floor(np.random.choice(np.arange(n2),n2,replace=False)*V/n2)

CVvalues= np.zeros((10,7))


##########################################################################KNN
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]
	
	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)

	krange=range(1,26,2)
	scoresflag=0
	kflag=0
	for k in krange:
		knn=KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train,Y_train)
		scores=knn.score(X_test,Y_test)
		if scores>scoresflag:
			scoresflag=scores
			kflag=k
	print('k-neighbors: ',kflag,' accuracy: ',scoresflag)
	knn_clf=KNeighborsClassifier(n_neighbors=kflag)
	knn_clf.fit(X_train, Y_train)
	
	CVvalues[i][0]=knn_clf.score(X_test,Y_test)#test classification rate

#####################################################################LOGISTIC
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]
	
	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)

	Crange=[0.001,0.01,0.1,1,10,100,1000]
	scoresflag=0
	Cflag=0
	for c in Crange:
		logis=LogisticRegression(solver="lbfgs", random_state=42, max_iter=5000,C=c)
		logis.fit(X_train,Y_train)
		scores=logis.score(X_test,Y_test)
		if scores>scoresflag:
			scoresflag=scores
			Cflag=c

	print('LogisticC: ',Cflag,' accuracy: ',scoresflag,' accuracy: ',scoresflag)
	logis_clf = LogisticRegression(solver="lbfgs", random_state=42, max_iter=5000,C=Cflag)
	logis_clf.fit(X_train, Y_train)
	
	CVvalues[i][1]=logis_clf.score(X_test,Y_test)#test classification rate


#####################################################################SVM-linear-sd
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]
	
	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)


	Crange=[0.001,0.01,0.1,1,3,6,9,12,20]
	scoresflagsd=0
	Cflagsd=0
	for c in Crange:
		svmsd=Pipeline([
		("scaler",StandardScaler()),
		("svm_clf",SVC(kernel="linear",C=c))])
		svmsd.fit(X_train,Y_train)
		scoressd=svmsd.score(X_test,Y_test)
		if scoressd>scoresflagsd:
			scoresflagsd=scoressd
			Cflagsd=c
	
	print('linearSVM_C: ',Cflagsd,' accuracy: ',scoresflagsd)
	SVMl_clf_sd = Pipeline([
		("scaler",StandardScaler()),
		("svm_clf",SVC(kernel="linear",C=Cflagsd))])
	SVMl_clf_sd.fit(X_train, Y_train)

	CVvalues[i][2]=SVMl_clf_sd.score(X_test,Y_test)#test classification rate


########################################################################SVM-poly-sd
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]

	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)

	
	Crange=[10**-3,10**-2,10**-1,0.5,1,5,10,100,1000]
	polydegs=[2,3,4]
	scoresflagsd=0
	Cflagsd=0
	polydeg=0
	for dee in polydegs:
		for c in Crange:
			svmpsd=Pipeline([
			("scaler",StandardScaler()),
			("svm_clf",SVC(kernel="poly",degree=dee,coef0=1,C=c))])
			svmpsd.fit(X_train,Y_train)
			scoressd=svmpsd.score(X_test,Y_test)
			if scoressd>scoresflagsd:
				scoresflagsd=scoressd
				Cflagsd=c
				polydeg=dee
	
	print('Poly_degree: ',polydeg,' Poly_C: ',Cflagsd,' accuracy: ',scoresflagsd)
	SVMp_clf_sd = Pipeline([
		("scaler",StandardScaler()),
		("svm_clf",SVC(kernel="poly",degree=polydeg,coef0=1,C=Cflagsd))])
	SVMp_clf_sd.fit(X_train, Y_train)

	CVvalues[i][3]=SVMp_clf_sd.score(X_test,Y_test)#test classification rate


########################################################################SVM-RBF-sd
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]

	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)

	
	Crange=[1,5,10,30,50,100,1000]
	gammarange=[0.00001,0.0001,0.001,0.01,0.1,0.5]
	scoresflagsd=0
	Cflagsd=0
	Gammaflagsd=0
	for c in Crange:
		for Gamma in gammarange:
			svmrbfsd=Pipeline([
			("scaler",StandardScaler()),
			("svm_clf",SVC(kernel="rbf",gamma=Gamma,C=c))])
			svmrbfsd.fit(X_train,Y_train)
			scoressd=svmrbfsd.score(X_test,Y_test)
			if scoressd>scoresflagsd:
				scoresflagsd=scoressd
				Cflagsd=c
				Gammaflagsd=Gamma
	
	print('RBF_gamma: ',Gammaflagsd,' RBF_C: ',Cflagsd,' accuracy: ',scoresflagsd)
	
	SVMrbfsd_clf = Pipeline([
			("scaler",StandardScaler()),
			("svm_clf",SVC(kernel="rbf",gamma=Gammaflagsd,C=Cflagsd))])
	SVMrbfsd_clf.fit(X_train, Y_train)

	CVvalues[i][4]=SVMrbfsd_clf.score(X_test,Y_test)#test classification rate


############################################################################################simpleNN-sd
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]

	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)

	Nsize=[5,20,50,[20,20],[50,50],100]
	alpharange=[0,0001,0.001,0.01,0.1,0.5,1]
	scoresflagsd=0
	Sflagsd=0
	Alphaflagsd=0
	for s in Nsize:
		for Alpha in alpharange:
			snnsd=Pipeline([
			("scaler",StandardScaler()),
			("snn_clf",MLPClassifier(alpha=Alpha,hidden_layer_sizes=s,max_iter=1000,random_state=42,activation='relu',warm_start=True))])
			snnsd.fit(X_train,Y_train)
			scoressd=snnsd.score(X_test,Y_test)
			if scoressd>scoresflagsd:
				scoresflagsd=scoressd
				Sflagsd=s
				Alphaflagsd=Alpha
	
	print("alpha: ",Alphaflagsd," layers: ",Sflagsd,' accuracy: ',scoresflagsd)
	
	snnsd_clf = Pipeline([
			("scaler",StandardScaler()),
			("snn_clf",MLPClassifier(alpha=Alphaflagsd,hidden_layer_sizes=Sflagsd,max_iter=1000,random_state=42,activation='relu',warm_start=True))])
	snnsd_clf.fit(X_train, Y_train)

	CVvalues[i][5]=snnsd_clf.score(X_test,Y_test)#test clasification rate

#############################################################################################RF
for i in range(V):
	train_set=eventdata.iloc[folds!=i,1:]
	train_labels=Originals.iloc[folds!=i,1]
	test_set=eventdata.iloc[folds==i,1:]
	test_labels=Originals.iloc[folds==i,1]

	X_train=np.array(train_set)
	Y_train=np.ravel(train_labels)

	X_test=np.array(test_set)
	Y_test=np.ravel(test_labels)

	Y_train=np.vectorize(dictionary.get)(Y_train)
	Y_test=np.vectorize(dictionary.get)(Y_test)


	estimators=[20,50,100,200]
	nodez=[2,4,5,7,9,11,15]
	maxfeat=[4,5,7,9,11,13] #FOR FEATURE GROUPS WITH HIGH NUMBER OF FEATURES
	#maxfeat=[4,5,7] #FOR FEATURE GROUPS WITH LOW NUMBER OF FEATURES
	scoresflag=0
	Nflag=0
	Featflag=0
	nestflag=0
	for nest in estimators:
		for n in nodez:
			for f in maxfeat:
				rf=RandomForestClassifier(n_estimators=nest, min_samples_split=n, max_features=f, random_state=42,n_jobs=3)
				rf.fit(X_train,Y_train)
				scores=rf.score(X_test,Y_test)
				if scores>scoresflag:
					scoresflag=scores
					Nflag=n
					Featflag=f
					nestflag=nest
	
	print("n_estimators: ",nestflag," min_nodes: ",Nflag," max_features: ",Featflag,' accuracy: ',scoresflag)
	
	RF_clf = RandomForestClassifier(n_estimators=nestflag, min_samples_split=Nflag, max_features=Featflag, random_state=42,n_jobs=5)
	RF_clf.fit(X_train, Y_train)

	CVvalues[i][6]=RF_clf.score(X_test,Y_test)#test classification rate


#######################POST PROCESS###########

CVerrors = 1-CVvalues

CVrelerr = np.multiply(CVerrors,(1/np.amin(CVerrors,axis=1)).reshape(-1,1))

CVsd = pd.DataFrame(CVerrors, columns=['knn', 'logis','SVM-lin-sd','SVM-poly-sd','SVM-RBF-sd','sNN','RF'])
boxplot=CVsd.boxplot(column=['knn', 'logis','SVM-lin-sd','SVM-poly-sd','SVM-RBF-sd','sNN','RF'],figsize=(12,5))

plt.show()

