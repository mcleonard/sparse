import os
from scipy import io
import numpy as np
from itertools import izip

def load_file(filepath, full=False):
    """ Load .mat files containing data from Vanessa's experiment 
    
        Returns a tuple (data, stims).  Data is the daq data as a numpy 
        structured array with fields signal, trigger, stimulus, and junk.  
        Stims also a structured array containing is the stimulus protocol.
        
        Set full=True to get the whole experiment protocol structure.
    """
    datadir, filename = os.path.split(filepath)
    if 'daq' in filename:
        daq_file = filepath
        exper_file = os.path.join(datadir, filename[4:])
    else:
        daq_file = os.path.join(datadir, 'daq_' + filename)
        exper_file = filepath
    
    data = load_daq(daq_file)
    exper, stims = load_exper(exper_file)
    
    if full:
        return data, stims, exper
    
    return data, stims
    
def load_daq(daq_file, verbose=False):
    """ Loads the daq information from the daq_*.mat file.  Returns a record
        array with 'signal', 'trigger', 'stimulus', and 'junk' fields.
    """
    daq = io.loadmat(daq_file)['daq']
    records = [('signal', 'f8'), ('trigger','f8'), 
               ('stimulus','f8'), ('junk','f8')]
    data = np.zeros(len(daq), dtype = records)
    for (col, _), row in izip(records, daq.T):
        data[col] = tuple(row)
    
    if verbose:
        print('{} loaded'.format(daq_file))
    
    return data
        
def load_exper(exper_file, verbose=False):
    """ Loads experiment protocol file and pulls out stimuli information.
        Returns the exper struct and a record array of stimulus information.
    """
    mat = io.loadmat(exper_file, chars_as_strings=True,
                     struct_as_record=False, squeeze_me=True)
    
    # Different protocols have different names.  They seem to be titled
    # 'exper_*', so I'm looking for a key with 'exper' in it.               
    exper_key = [ key for key in mat.viewkeys() if 'exper' in key ][0]
    exper = mat[exper_key]
    
    stims = exper.stimulusczar.param.stimuli.value
    
    # Get the fields and dtypes from stims
    stim = stims[0]
    fields = stim._fieldnames[:]
    dtypes = [ np.dtype(type(stim.__getattribute__(field))) 
               for field in fields if field != 'param']
    
    # Do the same for 'param' field
    if 'param' in fields:
        params = stim.param._fieldnames[:]
        fields.remove('param')
        fields.extend(params)
        param_dtypes = [ np.dtype(type(stim.param.__getattribute__(par))) 
                         for par in params ]
    dtypes = np.concatenate([dtypes, param_dtypes])
    
    # Switch all unicode fields to string fields.
    dtypes[np.where(dtypes == '<U0')]=np.dtype('S50')
    
    # Now make the record array and fill it.
    records = zip(fields, dtypes)
    stim_arr = np.zeros(len(stims), dtype=records)
    for i, stim in enumerate(stims):
        for name in stim_arr.dtype.names:
            if name in stim._fieldnames:
                stim_arr[name][i] = stim.__getattribute__(name)
            elif name in stim.param._fieldnames:
                stim_arr[name][i] = stim.param.__getattribute__(name)
    
    if verbose:
        print('{} loaded'.format(exper_file))
    
    return exper, stim_arr