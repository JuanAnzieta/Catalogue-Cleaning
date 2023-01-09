#!/usr/bin/env python3
import glob

listevent=glob.glob('./Events/*.mseed')
listevent.sort()

volcano_name='Cotopaxi'############MODIFY VOLCANO/CATALOG NAME HERE AS REQUIRED


h=open('./Catalogue/'+volcano_name+'_Catalogue_v0.csv','a+')#change name of volcano or dataset if required

k=0
for file in listevent:
	h.write(file[9:21]+','+file[19:21]+'\n')#modify here for the format of your files

h.close()

