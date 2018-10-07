"""
to load each group containing: 1 audio wav file
                               1 parameter json file
filenames will be contained within a csv file (1 group per line), or directly loaded from a directory
1. parse csvfile/directory. get path of all objects in each line and append to list
2. create a list of indices to draw samples (eg. index 52 = 4th wav file sample 390). this no. will also be the __len__
3. __getitem__:
    sample index
    load corresponding wav file
    pull out correct audio sample sequence
    load corresponding params
    convert audio to mu-law
    convert mu-law + params to tensor

@muhammad huzaifah 08/08/2018 
"""

import torch
import torch.utils.data as data

import os
import csv
from itertools import chain
import numpy as np
from natsort import natsorted
from librosa.core import load,get_duration
import math
import torchvision.transforms as transform

from paramManager import paramManager


def file2list(rootdir,extension):
	"""append to list all files in rootdir with given extension"""
	filelist = [os.path.join(rootdir, f) for f in os.listdir(rootdir) if f.endswith('.' + extension)]
	return filelist

def list2csv(filelist,csvfile):
	"""write each entry in filelist to csvfile, 1 entry per line. csvfile is a string of the csv filename"""
	with open(csvfile, 'w', newline='') as output:
		writer = csv.writer(output, lineterminator='\n')
		for val in filelist:
			writer.writerow([val])
			
def parse_playlist(csvfile):
	"""read in csvfile and parse line by line to a python list"""
	with open(csvfile, 'r', newline='') as f:
                reader = csv.reader(f)
                filelist = natsorted(list(chain.from_iterable(reader)))                
	#filelist = natsorted([line for line in pd.read_csv(csvfile,sep=',',header=None,chunksize=1)]) no longer using pandas
	return filelist
	
def check_duration(filelist):
	"""use librosa's get_duration method to find the duration of all files in filelist"""
	filedurations = [get_duration(filename=f) for f in filelist]
	assert filedurations.count(filedurations[0]) == len(filedurations), "File durations are not all the same!"
	return filedurations[0]
	
def dataset_properties(filelist,sr,seqLen):
	"""return several dataset parameters 
	input params: filelist - list of filenames that forms the dataset
				  sr - sample rate of the audio
				  seqLen - length of each data section measured in samples
	"""
	fileLen = len(filelist)                     #no of files in filelist
	fileDuration = check_duration(filelist)     #duration of each file in sec
	totalFileDuration = fileLen * fileDuration  #total duration of all files
	totalSamples = int(fileLen * fileDuration * sr)  #combined total no of samples
	srInSec = 1/sr                              #sampling rate in sec
	seqLenInSec = srInSec * seqLen              #length of 1 data sequence in sec
	return fileLen,fileDuration,totalFileDuration,totalSamples,srInSec,seqLenInSec
	
def create_sampling_index(totalSequences,stride):
	"""calculate total no. of data sections in dataset. Indices will later be used to draw sequences (eg. index 52 = 4th wav file sample 390)
	input params: totalSequences - combined total no of sequences in dataset (get from dataset_properties)
				  stride: shift in no. of samples between adjacent data sections. (seqLen - stride) samples will overlap between adjacent sequences.  
	"""	
	indexLen = totalSequences // stride
	return indexLen #list(range(indexLen))
	
def choose_sequence(index,fileDuration,srInSec,stride):
	"""identify the correct section of audio given the index sampled from indexLen"""
	timestamp = index * srInSec * stride
	chooseFileIndex = math.ceil(timestamp / fileDuration) - 1  #minus 1 since index start from 0
	startoffset = timestamp - chooseFileIndex * fileDuration   #will load at this start time  	
	return chooseFileIndex,startoffset
	
def load_sequence(filelist,chooseFileIndex,startoffset,seqLenInSec,seqLen,sr):
	"""load the correct section of audio. If len of audio < seqLen+1 (e.g. sections at the end of the file), then pad the rest with zeros
	We draw 1 sample more than seqLen so can take input=y[:-1] and target=y[1:]"""
	y,_ = load(filelist[chooseFileIndex],sr=sr,mono=True,offset=startoffset,duration=seqLenInSec+(1/sr))
	if len(y) < seqLen+1:
		y = np.concatenate((y,np.zeros(seqLen+1 - len(y))))  #pad sequence up to seqLen
	#sample = {'audio':y}
	assert len(y) == seqLen+1, str(len(y))
	return y


class AudioDataset(data.Dataset):
	 
	def __init__(self, sr, seqLen, stride, csvfile=None, datadir=None, extension=None, paramdir=None, prop=None, transform=None, param_transform = None, target_transform=None):

		if stride < 1:
			raise ValueError("stride has to be >= 1")
		
		if csvfile is not None:
			assert datadir is None, "Input either csvfile or data directory - Not both!"
			assert extension is None, "Do not input extension if csvfile is used!"
			self.filelist = parse_playlist(csvfile)
		elif datadir is not None:
			assert extension is not None, "Please input a file extension to use!"
			self.filelist = file2list(datadir,extension)
		else:
			raise ValueError("Please input either a csvfile or data directory to read from!")
			
		self.datadir = datadir
		self.paramdir= paramdir
		self.prop = prop
		self.fileLen,self.fileDuration,self.totalFileDuration,self.totalSamples,self.srInSec,self.seqLenInSec = dataset_properties(self.filelist,sr,seqLen)
		self.sr = sr
		self.seqLen = seqLen
		self.stride = stride
		self.transform = transform
		self.param_transform = param_transform
		self.target_transform = target_transform
		self.indexLen = create_sampling_index(self.totalSamples,self.stride)
            
	def __getitem__(self,index):
		chooseFileIndex,startoffset = choose_sequence(index+1,self.fileDuration,self.srInSec,self.stride)
		whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLenInSec,self.seqLen,self.sr).reshape(-1,1)
		sequence = whole_sequence[:-1]
		target = whole_sequence[1:]
		if self.transform is not None:
			input = self.transform(sequence)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.paramdir is not None:
			pm = paramManager.paramManager(self.datadir, self.paramdir) 
			params = pm.getParams(self.filelist[chooseFileIndex]) 
			#print("param",params["meta"]["filename"])
			paramdict = pm.resampleAllParams(params,self.seqLen,startoffset,startoffset+self.seqLenInSec,self.prop,verbose=False)
			if self.param_transform is not None:
				paramtensor = self.param_transform(paramdict)
				#print("param",paramtensor.shape)
				#input = {**input,**paramtensor}  #combine audio samples and parameters here
				input = torch.cat((input,paramtensor),1)  #input dim: (batch,seq,feature)	
		else:
			if self.transform is None:
				input = sequence
				
		return input,target
    
	def __len__(self):
		return self.indexLen 
		
	def rand_sample(self,index=None):
		if index is None:
			index = np.random.randint(self.indexLen)
		chooseFileIndex,startoffset = choose_sequence(index+1,self.fileDuration,self.srInSec,self.stride)
		whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLenInSec,self.seqLen,self.sr).reshape(-1,1)
		sequence = whole_sequence[:-1]
		return sequence
		
	
"""
from transforms import mulawnEncode,mulaw,array2tensor,dic2tensor	
sr = 16000
seqLen = 5
stride = 1

adataset = AudioDataset(sr,seqLen,stride,
						datadir="dataset",extension="wav",
						paramdir="dataparam",prop=['rmse','instID','midiPitch'],  
						transform=transform.Compose([mulawnEncode(256,0,1),array2tensor(torch.FloatTensor)]),
						param_transform=dic2tensor(torch.FloatTensor),
						target_transform=transform.Compose([mulaw(256),array2tensor(torch.LongTensor)]))

for i in range(len(adataset)):
    inp,target = adataset[i]
    print(inp)
    print(target)
    
    if i == 2:
        break 
"""

