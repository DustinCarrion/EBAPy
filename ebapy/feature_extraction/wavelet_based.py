import numpy as np


def verifyValue(value):
    if np.isinf(value) or np.isnan(value):
        return False
    return True


def minimum(signals):
    features = []
    for signal in signals:
        features.append(np.min(signal))
    return np.array(features)


def maximum(signals):
    features = []
    for signal in signals:
        features.append(np.max(signal))
    return np.array(features)


def mean(signals):
    features = []
    for signal in signals:
        features.append(np.mean(signal))
    return np.array(features)


def std(signals):
    features = []
    for signal in signals:
        features.append(np.std(signal))
    return np.array(features)


def variance(signals):
    features = []
    for signal in signals:
        features.append(np.var(signal))
    return np.array(features)


def median(signals):
    features = []
    for signal in signals:
        features.append(np.median(signal))
    return np.array(features)


def skewness(signals): 
    features = []
    for signal in signals:
        std = np.std(signal)
        if std == 0:
            return 0
        mean = np.mean(signal)
        median = np.median(signal)
        skewness_value = (3*(mean-median))/std
        if verifyValue(skewness_value):
            features.append(skewness_value)
        features.append(0)
    return np.array(features)


def energy(signals):
    features = []
    for signal in signals:
        features.append(np.sum(np.array(signal)**2))
    return np.array(features)


def entropy(signals): 
    features = []
    for signal in signals:
        entropy_value = 0
        for i in range(len(signal)):
            if verifyValue(np.log2(signal[i]**2)):
                entropy_value += (signal[i]**2)*(np.log2(signal[i]**2)) 
        if verifyValue(entropy_value):
            features.append(entropy_value)
        features.append(0)
    return np.array(features)


def relative_energy(signals):
    features = []
    energies = []
    for signal in signals:
        energies.append(np.sum(np.array(signal)**2))
    total_energy = np.sum(energies)
    for energy_value in energies:
        if total_energy == 0:
            features.append(0)
        else:
            features.append(energy_value/total_energy)
    return np.array(features)
    