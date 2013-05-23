import numpy as np

def process_data(ephys, exper):
    """ Takes the electrical recordings and experiment protocol and returns
        the spike times and stimulus information.  Spike times are locked to
        the onset for each stimulus.
    """
    _, peaks = detect_spikes(ephys['signal'])
    triggers = detect_triggers(ephys['trigger'])
    data = build_dframe(peaks, triggers, exper)

    return data

def detect_spikes(data, threshold=4):
    """ Detect spikes in data.  Returns spike waveform patches and peak samples. 
    """
    
    filtered = butter_filter(data, high = 4999, rate=10000) 
    peaks = crossings(filtered, medthresh(filtered, threshold))
    peaks = censor(peaks, 30)
    spikes, peaks = extract(data, peaks)
    
    return spikes, peaks
    
def detect_triggers(data):
    """ Detects the trigger onsets """
    triggers = censor(crossings(data, threshold=0.5), width=1000)
    _, tr_onsets = extract(data, triggers, patch_size=2)
    
    return tr_onsets
    
def build_dframe(spikes, triggers, exper, rate=10000, delay=70):
    """ Build a pandas DataFrame from the spike peak times, the trigger times,
        and the experiment records.
        
        Arguments
        ---------
        spikes : np.array : sample time stamps for each spike
        triggers : np.array : sample time stamps for each trigger onset
        exper : np structured array :
            experiment records, get this from sparse.io.load_data
        rate : int, float : sampling rate for the data, in samples per second
        delay : int, float :
            This is the delay between when the trigger starts and when the 
            stimulus starts.  I've found that on average it is 70 samples with 
            a std of 10 samples.  In general, you might want to check this from
            the data and set it accordingly.
        
    """
    from pandas import Series, DataFrame

    samp_rate = float(rate)
    
    times = [ lock_spikes(spikes, onset, delay) for onset in triggers ]
    time_stamps = Series(times)/samp_rate
    
    data = DataFrame(exper)
    data['spikes'] = time_stamps
    return data
    
def lock_spikes(spikes, onset, delay):
    """ Locks spike times to onset, returns the timelocked time stamps.
    """
    low_cutoff = (onset + delay - 10000) < spikes
    high_cutoff = spikes < (onset + delay + 11000)
    time_stamps = spikes[low_cutoff * high_cutoff] - onset - delay
    return time_stamps

def medthresh(data,threshold=4):
    """ A function that calculates the spike crossing threshold 
        based off the median value of the data.
    
    Arguments
    ---------
    data : your data
    threshold : the threshold multiplier
    """
    return threshold*np.median(np.abs(data)/0.6745)

def butter_filter(data, low=300, high=6000, rate=30000):
    """ Uses a 3-pole Butterworth filter to reduce the noise of data.
    
        Arguments
        ---------
        data : numpy array : data you want filtered
        low : int, float : low frequency rolloff
        high : int, float : high frequency rolloff
        rate : int, float : sample rate
    
    """
    import scipy.signal as sig
    
    if high > rate/2.:
        high = rate/2.-1
        print("High rolloff frequency can't be greater than the Nyquist \
               frequency.  Setting high to {}").format(high)

    filter_lo = low #Hz
    filter_hi = high #Hz
    samp_rate = float(rate)
    
    #Take the frequency, and divide by the Nyquist freq
    norm_lo = filter_lo/(samp_rate/2)
    norm_hi = filter_hi/(samp_rate/2)
    
    # Generate a 3-pole Butterworth filter
    b, a = sig.butter(3,[norm_lo,norm_hi], btype="bandpass");
    return sig.filtfilt(b,a,data)

def censor(data, width=30):
    """ This is used to insert a censored period in found threshold crossings.
        For instance, when you find a crossing in the signal, you don't
        want the next 0.5-1 ms, you just want the first crossing.
        
        Arguments
        ---------
        data : numpy array : data you want censored
        width : int : number of samples censored after a first crossing
    """
    edges = [data[0]]
    
    for sample in data:
        if sample > edges[-1] + width:
            edges.append(sample)
    return np.array(edges)
    
def crossings(data, threshold, polarity='pos'):
    """ Finds threshold crossings in data 
    
        Arguments:
        data : numpy array
        threshold : int, float : crossing threshold, always positive
        polarity : 'pos', 'neg', 'both' :
            'pos' : detects crossings for +threshold
            'neg' : detects crossings for -threshold 
            'both' : both + and - threshold
    """
    
    if type(data) != list:
        data = [data]
    peaks = []
    for chan in data:
        if polarity == 'neg' or polarity == 'both':
            below = np.where(chan<-threshold)[0]
            peaks.append(below[np.where(np.diff(below)==1)])
        elif polarity == 'pos' or polarity == 'both':
            above = np.where(chan>threshold)[0]
            peaks.append(above[np.where(np.diff(above)==1)])

    return np.concatenate(peaks)
    
def extract(data, peaks, patch_size=30, offset=0, polarity='pos'):
    """ Extract peaks from data based on sample values in peaks. 
    
        Arguments
        ---------
        patch_size : int : 
            number of samples to extract centered on peak + offset
        offset : int : 
            number of samples to offset the extracted patch from peak
        polarity : 'pos' or 'neg' :
            Set to 'pos' if your spikes have positive polarity
        
        Returns
        -------
        spikes : numpy array : N x patch_size array of extracted spikes
        peaks : numpy array : sample values for the peak of each spike
    """
    
    spikes, centered_peaks = [], []
    size = patch_size/2
    for peak in peaks:
        patch = data[peak-size:peak+size]
        if polarity == 'pos':
            peak_sample = patch.argmax()
        elif polarity == 'neg':
            peak_sample = patch.argmin()
        centered = peak-size+peak_sample+offset
        final_patch = data[centered-size:centered+size]
        centered_peaks.append(centered)
        spikes.append(final_patch)
        
    return np.array(spikes), np.array(centered_peaks)