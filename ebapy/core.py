from joblib import load as jLoad
from numpy import genfromtxt, array, savetxt, argmax
from numpy.random import randint
from pickle import load as pLoad
from mne.io import read_raw_edf
from os import listdir
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from ebapy.feature_extraction.wavelet_based import *
from ebapy.evaluation.performance_metrics import *

def create_eeg_matrix(directory, fix_length=False, length=None, row_channels=True):
    '''
    Creates an EEG matrix of dimensions t x c x l, where 't' are the number of trials, 'c' are the number of channels and 'l' is the recording length 

    Parameters
    ----------
    directory : str
        Name of the directory that contains the trials of a subject that must be joined in a single matrix.
    fix_length : bool, optional
        True if the trials have different lengths, False otherwise. The default is False.
    length : int, optional
        Length to be used in all trials. It should be specified only if 'fix_length' is True. The default is None.
    row_channels : bool, optional
        True if the rows of each trial corresponds to the channels, False otherwise. The default is True.

    Raises
    ------
    TypeError
        If the parameters or files have an unsupported type.
    RuntimeError
        If an error occurs that prevents the correct functioning.

    Returns
    -------
    eeg_matrix : ndarray
        EEG matrix of dimensions t x c x l.

    '''
    if type(directory) != str:
        raise TypeError(f"'directory' must be 'str', received: '{type(directory).__name__}'")
    
    if type(fix_length) != bool:
        raise TypeError(f"'fix_length' must be 'bool', received: '{type(fix_length).__name__}'")
    if fix_length:
        if (type(length) != int and type(length) != float) or (int(length) != length):
            raise TypeError(f"'length' must be 'int', received: '{type(length).__name__}'")
        if length < 1:
            raise TypeError(f"'length' must be greater than 0, received: {length}")
        length = int(length)
    
    if type(row_channels) != bool:
        raise TypeError(f"'row_channels' must be 'bool', received: '{type(row_channels).__name__}'")
        
    try:
        trials = listdir(directory)
    except:
        raise
    
    if len(trials) == 0:
        raise RuntimeError("Empty directory cannot be converted into EEG matrix")
    
    eeg_matrix = []
    trial_length = None
    for trial in trials:
        if "." not in trial:
            raise TypeError(f"File type of '{trial}' not found")
        
        file_type = trial.split(".")[-1]
        if file_type not in ["sav", "csv", "dat", "edf"]:
            raise TypeError(f"File type of '{trial}' not supported. Supported types: sav, csv, dat, and edf")
        
        try:
            if file_type == "sav":
                data = array(jLoad(f"{directory}/{trial}"))    
            elif file_type == "csv":
                data = genfromtxt(f"{directory}/{trial}", delimiter=",")
            elif file_type == "dat":
                data = array(pLoad(open(f"{directory}/{trial}", "rb")))                       
            elif file_type == "edf":
                raw_data = read_raw_edf(f"{directory}/{trial}")
                data = array(raw_data.get_data())       
        except:
            raise
    
        if not row_channels:
            data = data.T
        
        if fix_length:
            if len(data[0]) < length:
                raise RuntimeError(f"All trials must be at least {length} long")
            data = data[:,:length]
        else:
            if trial_length == None:
                trial_length = len(data[0])
            else:
                if len(data[0]) != trial_length:
                    raise RuntimeError("All trials must have the same length")
        
        eeg_matrix.append(data)

    return array(eeg_matrix)


def time_windowing(eeg_matrix, frequency, time_windows, start_type="begin", start_time=None, channels=None, verbose=True):
    '''
    Segments an EEG matrix into the specified time windows.

    Parameters
    ----------
    eeg_matrix : ndarray
        An EEG matrix of dimensions t x c x l, where 't' are the number of trials, 'c' are the number of channels and 'l' is the recording length.
    frequency : int
        Sampling frequency used for recording the EEG matrix.
    time_windows : list
        Time windows, in seconds, in which trials must be segmented.
    start_type : str, optional
        Method for selecting the start of the segmentation, it can be: 'begin' (starts the segmentation from the begining of the recording), 'random' (starts the segmentation from a random point of the recording), or 'custom' (allows the definition of the exact moment for starting the segmentation). The default is "begin".
    start_time : float, optional
        Specific point, in seconds, to begin the segmentation. It should be specified only if 'start_type' is 'custom'. The default is None.
    channels : list, optional
        List of channels to use for the segmentation, if None, then all channels used. The default is None.
    verbose : bool, optional
        True to receive information during the segmentation process, False otherwise. The default is True.

    Raises
    ------
    TypeError
        If the parameters have an unsupported type.
    RuntimeError
        If an error occurs that prevents the correct functioning.

    Returns
    -------
    time_segmented_matrices : list
        List with the corresponding segmented matrices to each time window.

    '''
    if type(eeg_matrix).__name__ != "ndarray" and type(eeg_matrix) != list:
        raise TypeError(f"'eeg_matrix' must be 'ndarray', received: '{type(eeg_matrix).__name__}'")
    if type(eeg_matrix) == list:
        eeg_matrix = array(eeg_matrix)
    if "float" not in eeg_matrix.dtype.name and "int" not in eeg_matrix.dtype.name:
        raise TypeError(f"All elements of 'eeg_matrix' must be numbers")
    if len(eeg_matrix.shape) != 3:
        raise TypeError(f"'eeg_matrix' must be an matrix of dimensions t x c x l, received: {eeg_matrix.shape}")
    
    if (type(frequency) != int and type(frequency) != float) or (int(frequency) != frequency):
        raise TypeError(f"'frequency' must be 'int', received: '{type(frequency).__name__}'")
    if frequency < 1:
        raise TypeError(f"'frequency' must be greater than 0, received: {frequency}")
    frequency = int(frequency)
    
    if type(time_windows).__name__ != "ndarray" and type(time_windows) != list:
        raise TypeError(f"'time_windows' must be list of numbers, received: '{type(time_windows).__name__}'")
    if type(time_windows) == list:
        time_windows = array(time_windows)
    if "float" not in time_windows.dtype.name and "int" not in time_windows.dtype.name:
        raise TypeError(f"All elements of 'time_windows' must be numbers")
    for time_window in time_windows:
        if not (time_window > 0):
            raise TypeError(f"All time windows must be greater than 0, received: {time_window}")
    
    if start_type not in ["begin", "random", "custom"]:
        raise TypeError(f"Invalid 'start_type' must be 'begin', 'random', or 'custom', received: '{start_type}'")
    if start_type == "custom":
        if type(start_time) != int and type(start_time) != float:
            raise TypeError(f"'start_time' must be a number, received: '{type(start_time).__name__}'")
        if not (start_time > 0):
            raise TypeError(f"'start_time' must be greater than 0, received: {start_time}")
        start = int(start_time*frequency)
    
    if type(channels) != list and channels != None:
        raise TypeError(f"'channels' must be list of integers, received: '{type(channels).__name__}'")
    if channels != None:
        for channel in channels:
            if type(channel) != int:
                raise TypeError(f"'channels' must be list of integers, received: '{type(channel).__name__}' inside the list")
            if channel < 0:
                raise TypeError(f"All channels must be greater or equal to 0, received: {channel}")
    
    if type(verbose) != bool:
        raise TypeError(f"'verbose' must be 'bool', received: '{type(verbose).__name__}'")
    
    recording_length = len(eeg_matrix[0,0])
    total_recording_time = recording_length/frequency
    for time in time_windows:
        if time > total_recording_time:
            raise RuntimeError(f"All time windows must be less than or equal to {total_recording_time} seconds because it is the maximum recording time available in the input EEG matrix")
        if start_type == "custom":
            if (time+start_time) > total_recording_time:
                raise RuntimeError(f"All time windows must be less than or equal to {total_recording_time-start_time} seconds because it is the maximum recording time available in the input EEG matrix minus the specified start time")
    
    time_segmented_matrices = []
    for time in time_windows:
        if verbose: print(f"***** Starting segmentation: {time} seconds *****")
        time_length = int(time*frequency)
        
        time_segmented_matrix = []
        for trial in range(len(eeg_matrix)):
            try:
                if channels == None:
                    if start_type == "begin":
                        segmented_trial = eeg_matrix[trial][:,:time_length]
                    elif start_type == "random":        
                        random_start =  randint(0, recording_length-time_length+1)
                        segmented_trial = eeg_matrix[trial][:,random_start:random_start+time_length]
                    else:
                        segmented_trial = eeg_matrix[trial][:,start:start+time_length]
                
                else:
                    segmented_trial = []
                    try:
                        if start_type == "random":        
                            random_start =  randint(0, recording_length-time_length+1)
                        for channel in channels:
                            if start_type == "begin":
                                segmented_trial.append(eeg_matrix[trial][channel,:time_length])
                            elif start_type == "random":        
                                segmented_trial.append(eeg_matrix[trial][channel,random_start:random_start+time_length])
                            else:
                                segmented_trial.append(eeg_matrix[trial][channel,start:start+time_length])
                    except:
                        raise
                    segmented_trial = array(segmented_trial)
            except:
                raise
            
            time_segmented_matrix.append(segmented_trial)
            if verbose: print(f"Trial {trial} segmented correctly")
        
        time_segmented_matrices.append(array(time_segmented_matrix))
        if verbose: print(f"{'*'*49}\n")
    
    return time_segmented_matrices


def extract_wavelet_features(signals, features):
    feature_vectors = []
    for feature in features:
        if feature == "min":
            feature_vectors.append(minimum(signals))
        elif feature == "max":
            feature_vectors.append(maximum(signals))
        elif feature == "mean":
            feature_vectors.append(mean(signals))
        elif feature == "std":
            feature_vectors.append(std(signals))
        elif feature == "var":
            feature_vectors.append(variance(signals))
        elif feature == "median":
            feature_vectors.append(median(signals))
        elif feature == "skewness":
            feature_vectors.append(skewness(signals))
        elif feature == "energy":
            feature_vectors.append(energy(signals))
        elif feature == "entropy":
            feature_vectors.append(entropy(entropy))
        elif feature == "rel_energy":
            feature_vectors.append(relative_energy(signals))
    return feature_vectors
        
     
def k_fold_cross_validation(data, classifiers, number_of_labels, metrics, classification_type, epochs=None):
    amount_classifiers = len(classifiers)
    confusion_matrices = []
        
    for i in range(amount_classifiers):
        confusion_matrices.append(np.zeros((number_of_labels,number_of_labels)))
        
    for fold in data:
        X_train = fold["train"]["data"]
        y_train = fold["train"]["labels"]
        X_test = fold["test"]["data"]
        y_test = fold["test"]["labels"]
        
        for i in range(amount_classifiers):
            if type(classifiers[i]).__name__ != "Sequential":
                clf = classifiers[i]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test) 
            else:
                model = classifiers[i]
                enc = OneHotEncoder(categories="auto")
                y_train_mlp = array(y_train).reshape(len(y_train), 1)
                y_train_mlp = enc.fit_transform(y_train_mlp).toarray()
                model.fit(X_train, y_train_mlp, batch_size=100, epochs=epochs, verbose=0)
                y_pred = model.predict(X_test) 
                y_pred = array(list(map(argmax, y_pred)))+1
                
            cm = confusion_matrix(y_test,y_pred)
            confusion_matrices[i]+=cm
    
    performance_metrics = []
    for i in range(amount_classifiers):
        if classification_type == "binary":    
            performance_metrics.append(binary_metrics(confusion_matrices[i], metrics))
        elif classification_type == "multi":
            performance_metrics.append(multiclass_metrics(confusion_matrices[i], metrics))
    
    return performance_metrics, confusion_matrices

        