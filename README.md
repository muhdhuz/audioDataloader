## AudioLoader

A Pytorch dataset class for raw audio data. Overrides torch.utils.data.Dataset.  
Loads audio files from a specified directory or a csv file containing file paths, together with the corresponding parameter file.  

**_getitem_ method**  
The dataloader ___getitem___ will iterate through the dataset. It returns each data sequence as a pytorch tensor consisting of the audio data and the various parameters in the order (sequence,features).  
e.g. for a data sequence 5 samples long, every row is a timestep, columns are (audio samples,param1,param2,param3):
```bash
input = 
tensor([[  0.5020,   0.1437,   0.0000,  70.0000],
        [  0.5020,   0.1437,   0.0000,  70.0000],
        [  0.5020,   0.1437,   0.0000,  70.0000],
        [  0.5059,   0.1437,   0.0000,  70.0000],
        [  0.5098,   0.1437,   0.0000,  70.0000]])
```
The target sequence has a single value per timestep:
```bash
target = 
tensor([[ 128],
        [ 128],
        [ 129],
        [ 130],
        [ 130]])
```		
**Parameters**  
The parameters files consist of json objects structed as nested python dicts. They are saved, loaded and modified via the [paramManager library](https://github.com/lonce/paramManager) by Lonce Wyse.
The paramManager found in this repository extends the original library with methods for resampling and updating params. To learn how to use ParamManager, try out the instructional ProcessFiles.ipynb.

**Running the AudioDataset class**
```bash
from transforms import mulawnEncode,mulaw,array2tensor,dic2tensor	
sr = 16000
seqLen = 5
stride = 1

adataset = AudioDataset(sr,seqLen,stride,
						datadir="dataset",extension="wav",
						paramdir="dataparam",prop=['rmse','instID','midiPitch'],  #parameters used for training can be specified here 
						transform=transform.Compose([mulawnEncode(256,0,1),array2tensor(torch.FloatTensor)]),
						param_transform=dic2tensor(torch.FloatTensor),
						target_transform=transform.Compose([mulaw(256),array2tensor(torch.LongTensor)]))

for i in range(len(adataset)):
    inp,target = adataset[i]
    print(inp)
    print(target)
    
    if i == 2:
        break 
```
**Dependencies**  
* pytorch 0.4.0  
* librosa 0.6.1  

**Authors**  
* Muhammad Huzaifah






