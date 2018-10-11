import numpy as np

import os  # for mkdir
from os import listdir, remove
from os.path import isfile, join
from scipy.interpolate import interp1d

import json

# --------------------------------------------------------
# 
# ---------------------------------------------------------

# A dictionary with some helper functions to enforce our format standard;
class pdict(dict) :
    '''A dictionary class with some helper functions to enforce our format standard:
    {'meta': {'filename' : foo,
               whatever_else},
     'param_name' : {'times' : [], 
                     'values' : [],
                     'units' : (categorical, frequency, amplitude, power, pitch, ...), default: None
                     'nvals' : number of potential values, defualt: 0 (means continuous real)
                     'minval: default: 0
                     'maxval: default: 1},
     ...
     }    
    
    '''
    def __init__(self, datafilename=None):
        '''Users should always use the datafilename.'''
        self['meta']={}
        if datafilename :
            self['meta']['filename']=datafilename
        
    def addMeta(self, prop, value) :
        '''Adds arbitrary properties to the meta data'''
        self['meta'][prop]=value
        
    def addParam(self, prop, times, values, units=None, nvals=0, minval=0, maxval=1) :
        '''Creates a parameter dictionary entry.'''
        self[prop]={}
        self[prop]['times']=times
        self[prop]['values']=values
        self[prop]['units']=units
        self[prop]['nvals']=nvals
        self[prop]['minval']=minval
        self[prop]['maxval']=maxval
        

# parameter files are json 
# Since only the dict gets json.dumped, we have to reconstruct when we load.
# If you are json.load'ing, pass the function as the object_hook parameter
# When this is passed as object_hook, all nested objects are processed with this function
def as_pdict(dct):
    if 'meta' in dct :   # if this is the 'root' ojbect, instantiate a new pdict class and set key vals
        foo=pdict()
        for key in dct : 
            foo[key]=dct[key]
        return foo
    else :
        return dct


# json doesn't know how to encode numpy data, so we convert them to python lists
# pass this class to json.dumps as the cls parameter
#     json.dumps(data, cls=NumpyEncoder)
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
#------------------------------------------------------------------------------------

# This is the class to import to manage parameter files
#
class paramManager() :
    def __init__(self, datapath, parampath) :
        '''Manages paramter files for the datafile stored in datapath. Creates parampath if it doesnt exist.'''
        self.datapath=datapath
        self.parampath=parampath
        if (not os.path.exists(parampath)) :
            os.makedirs(parampath)
            
    def filenames(self, dir) :
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    
        
    def initParamFiles(self, overwrite=False) : 
        '''Creates one paramter file in parampath for each data file in datapath.'''
        '''You can not addParameter()s until the parameter files exist.'''
        if ((os.path.exists(self.parampath)) and not overwrite) :
            print("{} already exists and overite is False; Not initializing".format(self.parampath))
            return
            
        for filename in self.filenames(self.datapath) :
            (shortname, extension) = os.path.splitext(filename)
            foo=pdict(filename)
            with open(self.parampath + '/' + shortname + '.params' , 'w') as file:
                file.write(json.dumps(foo, cls=NumpyEncoder, indent=4)) # use `json.loads` to do the reverse

                    
    def checkIntegrity(self) :
        '''Does a simple check for a 1-to-1 match between datafiles and paramfiles'''
        integrity=True
        
        # Is there a parameter file for every file in the datapath?
        for filename in os.listdir(self.datapath):
            (shortname, extension) = os.path.splitext(filename)
            if not os.path.isdir(self.datapath + "/" + filename) :
                if not os.path.isfile(self.parampath + '/' + shortname + '.params') :
                    print("{} does not exist".format(self.parampath + '/' + shortname + '.params'))
                    integrity=False
        
        # Is there a data file for every corresponding to the meta:filename stored in each param file?
        for filename in os.listdir(self.parampath):   
            with open(self.parampath + '/' + shortname + '.params') as fh:
                foo=json.load(fh)
                if not os.path.isfile(self.datapath+ "/" + foo['meta']['filename']) :
                    print("{} does not exist".format(self.datapath+ "/" + foo['meta']['filename']))
                    integrity=False
        return integrity

    # get the parameter data structure from full path named file
    def getParams(self, pfname) :
        '''Gest the pdict from the parmater file corresponding to the data file'''
        filename_w_ext = os.path.basename(pfname)
        shortname, ext = os.path.splitext(filename_w_ext)
        with open(self.parampath + '/' + shortname + '.params') as fh:
            params = json.load(fh, object_hook=as_pdict)
        return params
			
    
    def getFullPathNames(self, dir) :
        '''Returns a list of the full path names for all the data files in datapath.
        You will need the datapath/filename.ext in order to process files with other libraries.''' 
        flist=[]
        for filename in self.filenames(self.datapath) :
            flist.append(dir + '/' + filename)
        return flist
                    
    # add a parameter to the data sturcture and write the parameter file
    def addParam(self, pfname, prop, times, values, units=None, nvals=0, minval=0, maxval=1) :
        ''' Adds parameter data to the param file corresponding to a data file.
        pfname - data file
        prop - name of the parameter
        times - array of time points (in seconds) 
        values - array of values (must be equal in length to times)
        'units' : (categorical, frequency, amplitude, power, pitch, ...), default: None
        'nvals' : number of potential values, defualt: 0 (means continuous real)
        'minval: default: 0
        'maxval: default: 1
         '''
        filename_w_ext = os.path.basename(pfname)
        shortname, ext = os.path.splitext(filename_w_ext)
        
        params = self.getParams(pfname)
        #enforce 2 values for each parameter
        if len(times)<2 or len(values)<2:
            raise ValueError("each parameter has to have at least 2 values corresponding to its start and end")
			
        #add to the param data structure
        params.addParam(prop, times, values, units, nvals, minval, maxval)
        #save the modified structure to file
        with open(self.parampath + '/' + shortname + '.params' , 'w') as file:
                file.write(json.dumps(params, cls=NumpyEncoder, indent=4)) # use `json.loads` to do the reverse
				
				
    def resampleParam(self,params,prop,sr,timestart=None,timeend=None,verbose=False,overwrite=False):
        '''resample the chosen parameter by linear interpolation (scipy's interp1d). 
		Modifies the 'times' and 'values' entries but leaves others unchanged.
		Can resample the parameter for a chunk of audio by specifying timestart and timeend. Note: does not chunk the actual audio file. 
		Else the parameter will be resampled for the entire duration of the audio file.
        
		params - (loaded) json parameter file (output of getParams)
        prop - name of the parameter
        sr - new sampling rate i.e. len of new 'times' and 'values' list 
        timestart - start time corresponding to the audio timestamp in seconds
		timeend - end time corresponding to the audio timestamp in seconds
		verbose - prints out parameter before and after
        overwrite - overwrite the original parameter file with new values''' 
        
        #params = self.getParams(pfname)   #read existing parameter file into buffer
        if timestart is None:
            timestart = min(params[prop]['times'])
        if timeend is None:
            timeend = max(params[prop]['times'])
        if verbose:			
            print("--Data resampled from--")
            print("times:",params[prop]['times'])
            print("values:",params[prop]['values'])
        
        new_x = np.linspace(timestart, timeend, sr)
        try:
            new_y = interp1d(params[prop]['times'],params[prop]['values'],fill_value="extrapolate")(new_x)
        except ValueError:
            print(new_x,params[prop]['times'],params[prop]['values'])
        if verbose:
            print("--to--")
            print("times:",new_x)
            print("values:",new_y)
        #units = params[prop]['units']
        #nvals = params[prop]['nvals']
        #minval = params[prop]['minval']
        #maxval = params[prop]['maxval']

        if overwrite:
            params[prop]['times'] = new_x
            params[prop]['values'] = new_y
            filename_w_ext = params['meta']['filename']
            shortname, ext = os.path.splitext(filename_w_ext)		

            with open(self.parampath + '/' + shortname + '.params' , 'w') as file:
                file.write(json.dumps(params, cls=NumpyEncoder, indent=4))
				
        return new_x,new_y

    def resampleAllParams(self,params,sr,timestart=None,timeend=None,prop=None,verbose=False,overwrite=False):
        '''resample multiple parameters in parameter file using resampleParam method.
        prop contains the list of selected parameters. If None specified will default to all parameters (except meta).
        Will always ignore meta parameter.'''
        paramdict = {}
        if prop is None:
            prop = list(params.keys())
        for entry in prop:
            if entry != 'meta' and entry in params:
                if verbose:
                    print(entry)
                    _,value = self.resampleParam(params,entry,sr,timestart,timeend,verbose,overwrite)
                    print(' ')
                else:
                    _,value = self.resampleParam(params,entry,sr,timestart,timeend,verbose,overwrite)
                paramdict[str(entry)] = value
        return paramdict

            