import numpy as np
import torch
#
class mulawcore():
    #---  muval stuff
    def __init__(self, nvals):
        if nvals > 256 :
            raise ValueError('mulaw written to accommodate number of descrete value up to 256 only')
        self.nvals=nvals
        self.midpoint=(nvals-1.)/2.
        self.lognvals=np.log(nvals)


        #Brute force method to get the table of muvalues so we can index them with a one-hot vector
        #...just need enough numbers in linspace so we are sure to hit each muval at least once.
        #...12000 is adequate when the mutable length is 256, but goes up with 
        self.mutable=np.unique(self._decimate(self._float2mu(np.linspace(-1,1,num=12000, endpoint=False)))) 
        #print("mutable is of length = " + str(len(self.mutable)))

    #works on floats and nparrays of any size
    #note that this is continuous, not decimated
    def _float2mu(self,x) :
        return np.sign(x)*( (np.log(1+(self.nvals-1)*np.abs(x)))/self.lognvals)
    
    #works on floats and nparrays of any size
    #maps continuous floats in [-1,1] on to nvals equally spaced values in [-1,1]
    def _decimate(self, x) :
        return np.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.
    
    def _mu2float(self, x) :
        d=1/(self.nvals-1)
        y= np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) 
        return y



class mulaw(mulawcore):
    '''
    nvals is the number of quantization levels
    '''
    def __init__(self, nvals) :
        mulawcore.__init__(self, nvals)

    #-----------------------------------------------------------------------
    # -----  one-hot stuff
    # each row is converted from its onhot rep to its mu val
    def _onehot2mu(self, a) :
    	# looks like a can take a tensor, even a cuda tensor, and return a np array!
        argmax = np.argmax(a, 1)
        return   self.mutable[argmax]
    
    # maps mu-values to their mutable indicies
    def _mu2index(self, x) :
        return np.uint8((np.round((self.nvals-1)*(1+ self._decimate(x))/2)))
                        
                        
    def _mu2onehot1(self, m) :
        oh=np.zeros(self.nvals)
        oh[self._mu2index(m)]=1
        return oh
    
    def _mu2onehot(self, a) :
        siqlen = a.size
        oh=np.zeros((siqlen, self.nvals))
        for i in range(siqlen) :
            oh[i, : ] = self._mu2onehot1(a[i])
        return oh 

    def _float2index(self, a) : 
        return self._mu2index(self._decimate(self._float2mu(a)))

                        
    #-----------------------------------------------------------------------
    # *******************  for export only   ****************

    #array of floats to array of one-hot vectors
    def encode(self, a) :
        ilen=len(a)
        ar = np.zeros((ilen, self.nvals), dtype=np.uint8)
        idx=self._float2index(a)
        for i in range(ilen):
            # print(" se1 element " + str(i) + " for " + str(a[i]) + " has sets index " + str(self._float2index1(a[i])))
            ar[i][idx[i]] = 1
        return ar

    # oh is an tensor of one-hot vectors
    # returns: array of floats
    def decode(self, oh) :
        return self._mu2float(self._onehot2mu(oh))

    #input is an array of integers indexing the mutable
    def index2float(self, i) :
    	return self._mu2float(self.mutable[i])
    
    #index a float tensor of any number of dimension 
    # returns a ByteTensor of same shape as input
    # ------- should take an array of floats and return an array of floats
    def float2index(self, a) :
        ilen=len(a)
        #tensor = torch.zeros(ilen, 1, self.nvals).type(dtype)
        #eturn(torch.from_numpy(np.array(self._float2index(a))))
        return(self._float2index(a))
	
    def __call__(self,sample):
        return self.float2index(np.transpose([sample]))
    
    
##===========================================================================
##===========================================================================
# mulaw2 uses a dipole representation (two-element vector representing 
# a floating point value by coding the distance of teh value from each eand-point)
##===========================================================================

class mulawnEncode(mulawcore) :
    '''
    nvals is the number of quantization levels
    '''
    def __init__(self, qvals, paramType, numNodes) :
        mulawcore.__init__(self, qvals)
        self.paramType=paramType
        self.numNodes=numNodes
        self.halfn=paramType/2.

        # eithr single-value float, or spreading activation with even number sread (so sum activation is always 1)
        assert((paramType==0) and (numNodes==1)) or ((paramType>1) and (paramType%2==0) and (numNodes>1) ) , "Wrong combination of paramType and numNodes to mulawn"

    # *******************  for export only   ****************

    #array of floats to array of dipole vectors
    def __call__(self, sample) :
        a = sample
        ilen=len(a) # the sequence length
        ar = np.zeros((ilen, self.numNodes))
        #mu->decimation->normailize ; now we have a [0-1] float
        dnmuval=(1.+self._decimate(self._float2mu(a)))/2. #decimated to num quant vales, normalized to [0,1]
        
        if self.paramType==0 :
             ar[:,0]=dnmuval
        
        elif self.paramType==1 :
            pass # one-hot - should be using mulaw
        #distance activation within halfn
        elif(self.paramType%2==0) :
            floatIndex = dnmuval*(self.numNodes-1) # between 0 and max normedval
            d=np.abs([(floatIndex-i) for i in range(self.numNodes)]) # d is the distance from the float index
            ar = np.transpose(np.where( d < self.halfn, (1 -d/self.halfn)/self.halfn, 0))

        elif (True) :
            print("odd n-hots don't work")

        return ar

    def decode(self, sample) :
        ###NOT YET IMPLEMENTED!!!!!!!
        a = sample['audio']
        assert self.paramType <= 2, "haven't implemented decoding for nhot > 2"
        if self.paramType==0 :
            return self._mu2float(a[:,0]*2.-1.) 
        else :
            return self._mu2float(a[:,1]*2.-1.) #the 2nd dipole component is just the decimated and normalized muval


    def _cog1(self, a) :
        ''' Center of Gravity of the vector with weights at positions [0-1] inclusive.
        Input: 1D array
        Returns: 1-element array with center of gravity
        '''
        
        #sm = np.sum(a)
        sm=a.sum()
        if sm == 0 : 
            return [.5]
        idx=np.arange(len(a))

        #print("cog1 will return type " + str(type((1/(len(a)-1))*(idx*a).sum()/sm)))

        return (1/(len(a)-1))*(idx*a).sum()/sm

    def _cog2(self,a2):
        ''' Center of Gravity of the vector with weights at positions [0-1] inclusive.
        Input: 1- or 2-D array
        Returns an array with one number for each row of the input
        '''
        print("cog2 got type " + str(type(a2)))
        if self.paramType==0 :
            return a2.view(len(a2)) #"flatten" to 1D tensor

        #a2 = np.atleast_2d(a2)
        print("cog2 a2 type now " + str(type(a2)) +" and size " + str(a2.size()))
        #return np.array([self._cog1(a) for a in a2])
        return [self._cog1(a) for a in a2]


    
    #generalized for all spreading activation
    #def decodeN(self, a) :
    #    ''' Decodes rows of n-hot vectors to an array of floats noramlized in [-1, 1]
    #    '''
    #    return self._cog2(a)*2-1




##===========================================================================
##===========================================================================
# mulaw2 uses a dipole representation (two-element vector representing 
# a floating point value by coding the distance of teh value from each eand-point)
##===========================================================================
class mulaw2() :
    def __init__(self, nvals):
        if nvals > 256 :
            raise ValueError('mulaw written to accommodate number of descrete value up to 256 only')
        self.nvals=nvals
        self.midpoint=(nvals-1.)/2.
        self.lognvals=np.log(nvals)

        self.vlenth=2


        #Brute force method to get the table of muvalues so we can index them with a one-hot vector
        #...just need enough numbers in linspace so we are sure to hit each muval at least once.
        #...12000 is adequate when the mutable length is 256, but goes up with 
        self.mutable=np.unique(self._decimate(self._float2mu(np.linspace(-1,1,num=12000, endpoint=False)))) 
        #print("mutable is of length = " + str(len(self.mutable)))

        #works on floats and nparrays of any size
    def _float2mu(self,x) :
        return np.sign(x)*( (np.log(1+(self.nvals-1)*np.abs(x)))/self.lognvals)

    #works on floats and nparrays of any size
    #maps continuous floats in [-1,1] on to nvals equally spaced values in [-1,1]
    def _decimate(self, x) :
        return np.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.

    def _mu2float(self, x) :
        d=1/(self.nvals-1)
        y= np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) 
        return y


    # *******************  for export only   **************** 

    #array of floats to array of dipole vectors
    def encode(self, a) :
        ilen=len(a)
        ar = np.zeros((ilen, self.vlenth))
        dnmuval=(1.+self._decimate(self._float2mu(a)))/2. #decimated to num quant vales, normalized to [0,1]
        for i in range(ilen):
            ar[i][0]=1-dnmuval[i]
            ar[i][1]=dnmuval[i]
        return ar

    # oh is an array of one-hot vectors
    # returns: array of floats
    def decode(self, a) :
        return self._mu2float(a[:,1]*2.-1.) #the 2nd dipole component is just the decimated and normalized muval
 
 
class array2tensor():
    """Convert ndarrays in sample to Tensors. Samples are assumed to be python dics"""
    def __init__(self,dtype):
        self.dtype = dtype
	
    def __call__(self, sample):
        return torch.from_numpy(sample).type(self.dtype)
	
class dic2tensor():
    """Convert ndarrays in sample to Tensors. Samples are assumed to be python dics"""
    def __init__(self,dtype):
        self.dtype = dtype

    def __call__(self, sample):
        combined = np.stack([sample[i] for i in sample],axis=1)
        if len(sample) > 1:           
            tensor_sample = torch.squeeze(torch.from_numpy(combined).type(self.dtype))
        else:
            tensor_sample = torch.from_numpy(combined).type(self.dtype)		
        
        return tensor_sample

        