## AudioLoader

A Pytorch dataloader class for raw audio data. 
Loads audio files from a specified directory or a csv file containing file paths, together with the corresponding parameter file. 

**_getitem_ method**  
The dataloader ___getitem___ will iterate through the dataset. It returns each data sample as a python dict consisting of the audio data and the various parameters as pytorch tensors.
e.g. for a data section 5 samples long:
```bash
{'audio': tensor([[ 0.5020],
        [ 0.5020],
        [ 0.5020],
        [ 0.5059],
        [ 0.5098]]), 
 'rmse': tensor([ 0.1437,  0.1437,  0.1437,  0.1437,  0.1437]), 
 'instID': tensor([ 0.,  0.,  0.,  0.,  0.]), 
 'midiPitch': tensor([ 70.,  70.,  70.,  70.,  70.])}
```  
**Parameters**  
The parameters files consist of json objects structed as nested python dicts. They are saved, loaded and modified via the [paramManager library](https://github.com/lonce/paramManager) by Lonce Wyse.
The paramManager found in this repository extends the original library with methods for resampling and updating params. To learn how to use ParamManager, try out the instructional ProcessFiles.ipynb.

**Running the dataloader**
```bash
sr = 16000  
seqLen = 5  
stride = 1

adataset = AudioLoader(sr,seqLen,stride,datadir="dataset",paramdir="dataparam",extension="wav",
						transform=transform.Compose([mulawnEncode(256,0,1),ToTensor()]),param_transform=ToTensor())

for i in range(len(adataset)):
    sample = adataset[i]
    print(sample)
    
    if i == 3:
        break
```
**Dependencies**  
* pytorch  
* librosa  

**Authors**  
* Muhammad Huzaifah






