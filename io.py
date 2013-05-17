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
    
    # Load electrical recording data
    print daq_file
    daq = io.loadmat(daq_file)['daq']
    records = [('signal', 'f8'), ('trigger','f8'), 
               ('stimulus','f8'), ('junk','f8')]
    data = np.zeros(len(daq), dtype = records)
    for (col, _), row in izip(records, daq.T):
        data[col] = tuple(row)    
    
    # Load experiment protocol    
    exper = io.loadmat(exper_file, chars_as_strings=True,
                       struct_as_record=False, squeeze_me=True)
    exper = exper['exper_002']
    stimuli = exper.stimulusczar.param.stimuli.value
    records = [('stimulus', 'S20'), ('stimlength', 'i8'), ('amplitude', 'i8'),
               ('duration', 'i8'), ('frequency', 'f8'), ('ramp', 'i8')]
    stims = np.zeros(len(stimuli), dtype = records)

    for i, stim in enumerate(stimuli):
        stims[i] = (stim.type, stim.stimlength, stim.param.amplitude,
                    stim.param.duration, stim.param.frequency, stim.param.ramp)
    if full:
        return data, stims, exper
    
    return data, stims