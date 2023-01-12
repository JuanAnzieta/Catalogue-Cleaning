#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pandas as pd
import glob
from obspy import read
import re
from sklearn.metrics import confusion_matrix

def dft_calculator(readsignal,delta,dtype='PSD'):####modificar
	'''this function recieves a signal and calculates its DFT, returns the frequencies vector and PSD or frequency amplitude'''
	tracecopy=readsignal.copy()#create a copy of the trace
	xH = np.log(len(tracecopy.data))/np.log(2)#for fft to work we need to work with vector lengths of powers of two
	yH = np.ceil(xH) #approximates to the higher power of two
	dataH = np.zeros(2**int(yH))#gives back a full of zeroes vector of length power of two
	NSH1=tracecopy.data#extract only data to input in a power of two vector
	dataH[:len(NSH1)] = NSH1#fills the full of zeroes vector with the data until it can, then leaves zeroes
	spectrumH2 = fft.rfft(dataH) #calculates the DFT of the data for real values only since data is real -faster-
	freqH2 = fft.rfftfreq(len(dataH),d=delta)#calculates the frequency vector for the signal
	
	amplitude = abs(spectrumH2)
	sdensity = amplitude**2

	if dtype == 'PSD':
		return (freqH2,sdensity)
	if dtype == 'Amp':
		return (freqH2,amplitude)


volcano_name='Cotopaxi'############MODIFY VOLCANO/CATALOG NAME HERE AS REQUIRED

version='v0'#############MODIFY THIS ACCORDING TO THE CLEANING ITERATION


listlabels=glob.glob('./Predictions/*.csv')
listlabels.sort()

Originals=pd.read_csv("./Catalogue/"+volcano_name+"_Catalogue_"+version+".csv",header=None)
original_labels=np.array(Originals.iloc[:,1])

event_types=['LP','VT']


predicted_labels=pd.read_csv(listlabels[0],header=None)#modify depending on the listlabels(end or begining of list)
models_names=['Original_Labels']
for k in range(np.shape(predicted_labels)[1]-1):
	m=re.search(r'/|\\',listlabels[0])
	n=re.search(r'\\',listlabels[0])
	if len(predicted_labels)==len(Originals):
		models_names.append(listlabels[0][max(m.end(),n.end())+21:-4])
	if len(predicted_labels)-len(Originals)==1:
		models_names.append(listlabels[0][max(m.end(),n.end())+21:-4]+'-'+predicted_labels.iloc[0,k+1])
	if abs(len(predicted_labels)-len(Originals)>1):
		print('check dimensions of files in labels folder')

for labfile in listlabels[1:]:#modify depending on which is the original label (end or begining of list)
	temp=pd.read_csv(labfile,header=None)
	for i in range(np.shape(temp)[1]-1):
		m=re.search(r'/|\\',labfile)
		n=re.search(r'\\',labfile)
		if len(temp)-len(Originals)==1:
			models_names.append(labfile[max(m.end(),n.end())+21:-4]+'-'+temp.iloc[0,i+1])
		if len(temp)==len(Originals):
			models_names.append(labfile[max(m.end(),n.end())+21:-4])
		if abs(len(temp)-len(Originals)>1):
			print('check dimensions of files in labels folder')
	if len(temp)==len(predicted_labels):
		predicted_labels=pd.concat([predicted_labels,temp.iloc[:,1:]],axis=1)
	if len(temp)-len(predicted_labels)==1:
		predicted_labels=pd.concat([predicted_labels,temp.iloc[1:,1:].reset_index(drop=True)],axis=1)
	if abs(len(temp)-len(Originals)>1):
		print('check dimensions of files in labels folder')

#####################################PREDICTION ACCURACIES################################
pred_arr=np.array(predicted_labels.iloc[1:,1:])

####pred_matrix is the matrix of colors
truelabs_arr=np.array(original_labels)
pred_matrix=np.zeros((np.shape(pred_arr)[0],np.shape(pred_arr)[1]+1,3))

for col in range(np.shape(pred_arr)[1]):####prediction coloring
	for evl in range(np.shape(pred_arr)[0]):
		if pred_arr[evl,col]==truelabs_arr[evl]:
			pred_matrix[evl,col+1]=[0,1,0]
		else:
			pred_matrix[evl,col+1]=[1,0,0]

for row in range(len(truelabs_arr)):####originals coloring
	if truelabs_arr[row]==event_types[0]:
		pred_matrix[row,0]=[0,0,1]
	elif truelabs_arr[row]==event_types[1]:
		pred_matrix[row,0]=[1,1,0]

###############model predictions plotting
Nmodels=np.shape(pred_matrix)[1]
for j in range(int(len(truelabs_arr)/(Nmodels))):
	print(j)
	plt.figure(figsize=(5.9,8.2))
	plt.imshow(pred_matrix[j*Nmodels:j*Nmodels+Nmodels,:])

	ax = plt.gca();
	#major ticks
	ax.set_xticks(np.arange(0, Nmodels, 1))
	#minor ticks
	ax.set_xticks(np.arange(-.5, Nmodels, 1), minor=True)
	ax.set_yticks(np.arange(-.5, Nmodels, 1), minor=True)

	# Gridlines based on minor ticks
	ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

	#for major ticks on top
	ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,which='major')

	# for minor ticks to dissapear
	ax.tick_params(which='minor',size=0.0)

	###setting the names of the models
	names2=models_names
	ax.set_xticklabels(names2, rotation=90)

	plt.xlabel(volcano_name+' models')
	plt.ylabel(volcano_name+' events')
	plt.yticks([], [])
	plt.tight_layout()
	plt.show()
	plt.close()

plt.figure(figsize=(5.9,8.2))
plt.imshow(pred_matrix[Nmodels*int(len(truelabs_arr)/Nmodels):,:])
ax = plt.gca();
# Major ticks
ax.set_xticks(np.arange(0, Nmodels, 1))
# Minor ticks
ax.set_xticks(np.arange(-.5, Nmodels, 1), minor=True)
ax.set_yticks(np.arange(-.5, Nmodels, 1), minor=True)
# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
#for major ticks on top
ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,which='major')
# for minor ticks to dissapear
ax.tick_params(which='minor',size=0.0)
###setting the names of the models
names2=models_names
ax.set_xticklabels(names2, rotation=90)
plt.xlabel(volcano_name+' models')
plt.ylabel(volcano_name+' events')
plt.yticks([], [])
plt.show()
plt.close()


######event quality based on agreements
event_scores=np.zeros(len(truelabs_arr))
for i in range(len(event_scores)):
	event_sum=0
	for j in range(1,np.shape(pred_matrix)[1]):
		event_sum=event_sum+pred_matrix[i,j,1]
	event_scores[i]=event_sum/(np.shape(pred_matrix)[1]-1)

event_quality_df=pd.concat([Originals,pd.DataFrame(event_scores)],axis=1)

consistency_matrix=np.zeros((5,len(event_types)+1),dtype='U17')
consistency_matrix[0,1:]=event_types
consistency_matrix[1,0]='mean'
consistency_matrix[2,0]='sd'
consistency_matrix[3,0]='max'
consistency_matrix[4,0]='min'

for k in range(len(event_types)):
	consistency_matrix[1,k+1]=str(100*event_quality_df[event_quality_df.iloc[:,1]==event_types[k]].mean().iloc[0])
	consistency_matrix[2,k+1]=str(100*event_quality_df[event_quality_df.iloc[:,1]==event_types[k]].std().iloc[0])
	consistency_matrix[3,k+1]=str(100*max(event_quality_df[event_quality_df.iloc[:,1]==event_types[k]].iloc[:,2]))
	consistency_matrix[4,k+1]=str(100*min(event_quality_df[event_quality_df.iloc[:,1]==event_types[k]].iloc[:,2]))

print(consistency_matrix)
const_df=pd.DataFrame(consistency_matrix)
#const_df.to_csv('./Statistics/'+volcano_name+'_Class_Consistency_Stats_'+version+'.csv',index=False,header=False)
##############global matrix
global_matrix=np.zeros((4,2),dtype='U17')
global_matrix[0,0]='mean'
global_matrix[1,0]='sd'
global_matrix[2,0]='max'
global_matrix[3,0]='min'
global_matrix[0,1]=str(100*event_quality_df.iloc[:,2].mean())
global_matrix[1,1]=str(100*event_quality_df.iloc[:,2].std())
global_matrix[2,1]=str(100*max(event_quality_df.iloc[:,2]))
global_matrix[3,1]=str(100*min(event_quality_df.iloc[:,2]))
glob_df=pd.DataFrame(global_matrix)
#glob_df.to_csv('./Statistics/'+volcano_name+'_Global_Consistency_Stats_'+version+'.csv',index=False,header=False)

#########HISTOGRAMS

all_LPs=event_quality_df.loc[event_quality_df.iloc[:,1]==event_types[0]]
all_LPs.iloc[:,2].plot.hist(bins=15,rwidth=0.9,color='blue',align='right',log=True)
plt.xlabel('Consistency')
plt.ylabel('Log. Frequency')
plt.title(volcano_name+' '+event_types[0]+'s')
plt.show()

all_VTs=event_quality_df.loc[event_quality_df.iloc[:,1]==event_types[1]]
all_VTs.iloc[:,2].plot.hist(bins=15,rwidth=0.9,color='yellow',align='right',log=False)
plt.xlabel('Consistency')
plt.ylabel('Frequency')
plt.title(volcano_name+' '+event_types[1]+'s')
plt.show()

event_quality_df.iloc[:,2].plot.hist(bins=15,rwidth=0.9,color='#86bf91',align='right',log=True)
plt.xlabel('Consistency')
plt.ylabel('Log. Frequency')
plt.title(volcano_name+' all events')
plt.show()


###################BAD/GOOD EVENTS LIST

lista_bad_LP=event_quality_df.loc[(event_quality_df.iloc[:,1]==event_types[0]) & (event_quality_df.iloc[:,2]<0.67)]
lista_bad_VT=event_quality_df.loc[(event_quality_df.iloc[:,1]==event_types[1]) & (event_quality_df.iloc[:,2]<0.25)]

LP_array_bad=np.array(lista_bad_LP)
VT_array_bad=np.array(lista_bad_VT)
for event in LP_array_bad:
#for event in VT_array_bad:
	trace=read('./Events/'+event[0]+'.mseed')
	
	
	freqs,amplitudes=dft_calculator(trace[0].data,trace[0].stats.delta,'Amp')
	fig, axs = plt.subplots(3,1,figsize=(5.7,5))
	axs[0].plot(np.array(range(len(trace[0])))/trace[0].stats.sampling_rate,trace[0].data,'k')
	axs[1].sharex(axs[0])
	axs[1].specgram(trace[0],Fs=trace[0].stats.sampling_rate,cmap='gist_stern',scale='linear',mode='magnitude')
	axs[2].plot(freqs,amplitudes,'k')
	axs[2].set_xlim([0,trace[0].stats.sampling_rate/2])
	axs[0].text(0.7,0.7,'score: '+str(round(event[2],2)),transform=axs[0].transAxes)
	axs[0].set_xlim([0,len(trace[0])/trace[0].stats.sampling_rate])
	plt.suptitle(event[0])
	#plt.savefig('./Figures/VT_check/'+event[0]+'_'+str(round(event[2],4))+'.png')
	plt.show()
	plt.close()
	

lista_good_LP=event_quality_df.loc[(event_quality_df.iloc[:,1]=='LP') & (event_quality_df.iloc[:,2]>0.99)]
lista_good_VT=event_quality_df.loc[(event_quality_df.iloc[:,1]=='VT') & (event_quality_df.iloc[:,2]>0.70)]

LP_array_good=np.array(lista_good_LP)
VT_array_good=np.array(lista_good_VT)
j=0
#for event in LP_array_good:
for event in VT_array_good:
	trace=read('./Events/'+event[0]+'.mseed')
	
	j=j+1

	freqs,amplitudes=dft_calculator(trace[0].data,trace[0].stats.delta,'Amp')
	fig, axs = plt.subplots(3,1,figsize=(5.7,5))
	axs[0].plot(np.array(range(len(trace[0])))/trace[0].stats.sampling_rate,trace[0].data,'k')
	axs[1].sharex(axs[0])
	axs[1].specgram(trace[0],Fs=trace[0].stats.sampling_rate,cmap='gist_stern',scale='linear',mode='magnitude')
	axs[2].plot(freqs,amplitudes,'k')
	axs[2].set_xlim([0,trace[0].stats.sampling_rate/2])
	axs[0].text(0.7,0.7,'score: '+str(round(event[2],2)),transform=axs[0].transAxes)
	axs[0].set_xlim([0,len(trace[0])/trace[0].stats.sampling_rate])
	plt.suptitle(event[0])
	plt.show()
	plt.close()
	#if j==11:#uncomment this if too many good events 
	#	break


