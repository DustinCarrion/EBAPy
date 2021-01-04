from pywt import Wavelet
from math import floor, ceil
from numpy import concatenate, flipud, zeros, convolve, array

def padding_symmetric(signal, size=8):
    '''
    Applies a symmetric padding of the specified size to the input signal.

    Parameters
    ----------
    signal : ndarray
        The signal to be padded.
    size : int, optional
        The size of the padding which corresponds to the size of the filter. The default is 8.

    Returns
    -------
    padded_signal : ndarray
        Padded signal.

    '''
    
    padded_signal = concatenate([flipud(signal[:size]), signal, flipud(signal[-size:])])
    return padded_signal


def restore_signal(signal, reconstruction_filter, real_len):
    '''
    Restores the signal to its original size using the reconstruction filter.

    Parameters
    ----------
    signal : ndarray
        The signal to be restored.
    reconstruction_filter : list
        The reconstruction filter to be used for restoring the signal.
    real_len : int
        Real length of the signal.

    Returns
    -------
    restored_signal : ndarray
        Restored signal of the specified length.

    '''
    restored_signal = zeros(2 * len(signal) + 1)
    for i in range(len(signal)):
        restored_signal[i*2+1] = signal[i]
    restored_signal = convolve(restored_signal, reconstruction_filter)
    restored_len = len(restored_signal)
    exceed_len = (restored_len - real_len) / 2
    restored_signal = restored_signal[int(floor(exceed_len)):(restored_len - int(ceil(exceed_len)))]
    return restored_signal


def DWT(signal, level=3, mother_wavelet='db4'):
    '''
    Applies a Discrete Wavelet Transform to the signal.

    Parameters
    ----------
    signal : ndarray
        The signal on which the DWT will be applied.
    level : int, optional
        The decomposition levels for the DWT. The default is 3.
    mother_wavelet : str, optional
        The mother wavelet that it is going to be used in the DWT. The default is "db4".

    Returns
    -------
    restored_approx_coeff : list
        Restored approximations coefficients.
    restored_detail_coeff : list
        Restored detail coefficients.

    '''
    if type(signal).__name__ != "ndarray" and type(signal) != list:
        raise TypeError(f"'signal' must be 'ndarray', received: '{type(signal).__name__}'")
    if type(signal) == list:
        signal = array(signal)
    if "float" not in signal.dtype.name and "int" not in signal.dtype.name:
        raise TypeError(f"All elements of 'signal' must be numbers")
           
    if type(level) != int:
        raise TypeError(f"'level' must be 'int', received: '{type(level).__name__}'")
    if level < 1:
        raise TypeError(f"'level' must be greater than 0, received: {level}")
        
    if mother_wavelet not in ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'dmey', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'shan', 'fbsp', 'cmor']:
        raise TypeError(f"Invalid 'mother_wavelet' must be 'haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'dmey', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'shan', 'fbsp', or 'cmor', received: '{mother_wavelet}'")
        
    original_len = len(signal)
    approx_coeff = []
    detail_coeff = []
    wavelet = Wavelet(mother_wavelet)
    low_filter = wavelet.dec_lo
    high_filter = wavelet.dec_hi
    filter_size = len(low_filter)
    try:
        for _ in range(level):
            padded_signal = padding_symmetric(signal, filter_size)
            low_pass_filtered_signal = convolve(padded_signal, low_filter)[filter_size:(2*filter_size)+len(signal)-1] 
            low_pass_filtered_signal = low_pass_filtered_signal[1:len(low_pass_filtered_signal):2]
            high_pass_filtered_signal = convolve(padded_signal, high_filter)[filter_size:filter_size+len(signal)+filter_size-1]
            high_pass_filtered_signal = high_pass_filtered_signal[1:len(high_pass_filtered_signal):2]
            approx_coeff.append(low_pass_filtered_signal)
            detail_coeff.append(high_pass_filtered_signal)
            signal = low_pass_filtered_signal
    except:
        raise
    low_reconstruction_filter = wavelet.rec_lo
    high_reconstruction_filter = wavelet.rec_hi
    real_lengths = []
    for i in range(level-2,-1,-1):
        real_lengths.append(len(approx_coeff[i]))
    real_lengths.append(original_len)
    restored_approx_coeff = []
    for i in range(level):
        restored_signal = restore_signal(approx_coeff[i], low_reconstruction_filter, real_lengths[level-1-i])
        for j in range(i):
            restored_signal = restore_signal(restored_signal, low_reconstruction_filter, real_lengths[level-i+j])
        restored_approx_coeff.append(restored_signal)
    restored_detail_coeff = []
    for i in range(level):
        restored_signal = restore_signal(detail_coeff[i], high_reconstruction_filter, real_lengths[level-1-i])
        for j in range(i):
            restored_signal = restore_signal(restored_signal, high_reconstruction_filter, real_lengths[level-i+j])
        restored_detail_coeff.append(restored_signal)
    return restored_approx_coeff, restored_detail_coeff 
