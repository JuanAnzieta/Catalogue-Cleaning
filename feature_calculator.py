#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import detrend
from scipy.signal import hilbert
import glob
from obspy import read
import librosa

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


def watson_feat(tr,delta):
	dataTD=tr/max(abs(tr))
	sdTD=np.std(dataTD)
	kurtTD=kurtosis(dataTD)
	skewTD=skew(dataTD)
	
	freqs,amplitudes=dft_calculator(tr,delta,'Amp')
	namplitudes=amplitudes/max(amplitudes)
	cummamplitudes=np.cumsum(namplitudes)
	cummamplitudes=cummamplitudes/max(cummamplitudes)
	
	peakFD=freqs[amplitudes==max(amplitudes)]
	medianFD=freqs[(np.abs(cummamplitudes - 0.5)).argmin()]
	skewFD=skew(namplitudes)
	crosses=np.argwhere(np.diff(np.sign(namplitudes - 0.7071))).flatten()
	qualityfactor=peakFD/(freqs[crosses][freqs[crosses]>=peakFD][0]-freqs[crosses][freqs[crosses]<peakFD][-1])

	feat_vector=np.array([sdTD,kurtTD,skewTD,peakFD[0],medianFD,skewFD,qualityfactor[0]])
	return(feat_vector)

def titos_feat(tr,delta):
	tr=tr/max(abs(tr))
	librosacoefs1=librosa.lpc(tr[0:int(len(tr)/3)],order=5)
	librosacoefs2=librosa.lpc(tr[int(len(tr)/3):2*int(len(tr)/3)],order=5)
	librosacoefs3=librosa.lpc(tr[2*int(len(tr)/3):],order=5)

	cummtr=np.cumsum(abs(tr))
	cummtr=cummtr/max(cummtr)
	reltime=np.arange(0,len(cummtr))
	tr20=reltime[(np.abs(cummtr - 0.2)).argmin()]
	tr50=reltime[(np.abs(cummtr - 0.5)).argmin()]
	tr80=reltime[(np.abs(cummtr - 0.8)).argmin()]

	freqs,amplitudes=dft_calculator(tr,delta,'Amp')
	namplitudes=amplitudes/max(amplitudes)#normalizing FD
	cummamplitudes=np.cumsum(namplitudes)
	cummamplitudes=cummamplitudes/max(cummamplitudes)
	ampli20=freqs[(np.abs(cummamplitudes - 0.2)).argmin()]
	ampli50=freqs[(np.abs(cummamplitudes - 0.5)).argmin()]
	ampli80=freqs[(np.abs(cummamplitudes - 0.8)).argmin()]

	feat_vector=np.concatenate(([tr20,tr50,tr80,ampli20,ampli50,ampli80],librosacoefs1[1:],librosacoefs2[1:],librosacoefs3[1:]))
	return(feat_vector)


def sotoaj_feat(tr,delta):
	tr=tr/max(abs(tr))
	freqs,amplitudes=dft_calculator(tr,delta,'Amp')
	namplitudes=amplitudes/max(amplitudes)
	interval_size=int(len(namplitudes)/(2*7))
	subamplis_means=np.array([np.mean(namplitudes[2+i*interval_size:2+(i+1)*interval_size]) for i in range(7)])

	envelopets=abs(hilbert(np.cumsum(abs(tr))/max(np.cumsum(abs(tr)))-np.arange(len(tr))/len(tr)))
	rescaledtimes=2*np.arange(len(tr))/len(tr) - 1
	coefts=np.polynomial.legendre.legfit(rescaledtimes,envelopets,7)

	duration=len(tr)
	feat_vector=np.concatenate(([duration],coefts,subamplis_means))
	return(feat_vector)

listevent=glob.glob('./Events/*.mseed')
listevent.sort()

volcano_name='Cotopaxi'############MODIFY VOLCANO/CATALOG NAME HERE AS REQUIRED

h=open('./Features/'+volcano_name+'_features_watson.csv','a+')#change name of volcano or dataset if required
q=open('./Features/'+volcano_name+'_features_titos.csv','a+')#change name of volcano or dataset if required
p=open('./Features/'+volcano_name+'_features_sotoaj.csv','a+')#change name of volcano or dataset if required

k=0
for file in listevent:
	trace=read(file)
	event_name=file[:-5]#modify this line according to your events names
	trace.detrend()
	trace.filter('highpass',freq=0.05)
	dt=trace[0].stats.sampling_rate
	delta=trace[0].stats.delta
	raw_dataB=trace[0].data[int(10*dt):int(-10*dt)]#####modify if files do not have added time before and after event
	feats3=watson_feat(raw_dataB,delta)
	feats4=titos_feat(raw_dataB,delta)
	feats7=sotoaj_feat(raw_dataB,delta)
	h.write('%s,%f,%f,%f,%f,%f,%f,%f\n' % (event_name,feats3[0],feats3[1],feats3[2],feats3[3],feats3[4],feats3[5],feats3[6]))
	q.write('%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (event_name,feats4[0],feats4[1],feats4[2],feats4[3],feats4[4],feats4[5],feats4[6],feats4[7],feats4[8],feats4[9],feats4[10],feats4[11],feats4[12],feats4[13],feats4[14],feats4[15],feats4[16],feats4[17],feats4[18],feats4[19],feats4[20]))
	p.write('%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (event_name,feats7[0],feats7[1],feats7[2],feats7[3],feats7[4],feats7[5],feats7[6],feats7[7],feats7[8],feats7[9],feats7[10],feats7[11],feats7[12],feats7[13],feats7[14],feats7[15]))
	k=k+1
	print('progress:',100*k/len(listevent))

h.close()
q.close()
p.close()