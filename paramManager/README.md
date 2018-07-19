# paramManager

A library class for managing json paramter files.
paramManager writes and reads parameter files for data file sets, one structured parameter dictionary per data file.

This is what parameter files look like:
~~~~
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
~~~~
The parameter files can then be read by dataLoaders, etc. 


## Getting Started
~~~~
from paramManager import paramManager

pm=paramManager(datapath, parampath)

pm.addParam(fname, "pitch", time_array, value_array, units="frequency", nvals=0, minval=0, maxval=1) 

foo=pm.getParams(filename)
pitchparam=fooparams['pitch']

plt.figure()
plt.title(title)
plt.plot(pitchparam['times'], pitchparam['values'])
~~~~

### Prerequisites

Nothing special



## Authors

* **Lonce Wyse**  [lonce.org](http://lonce.org)





