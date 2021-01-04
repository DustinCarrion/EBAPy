import sys
if ("../" not in sys.path):
    sys.path.insert(1, '../')

from ebapy.core import *
from ebapy.preprocessing.dwt import DWT
import numpy as np
from ebapy.helpers.data_split import split_data, create_folds
from ebapy.optimization.grid_search import *

#------------CODE PARAMETERS------------
labels = [1, 2, 3, 4, 5]
recording_length = 320
frequency = 128
time_windows = [0.25, 1.75]
channels = [0, 1, 4, 5, 8, 9]
decomposition_levels = [2, 3]
mother_wavelet = 'db4'
detail_coefficients = {2: [1,2], 3: [1,2,3]}
approximation_coefficients = {2: [2], 3: [3]}
features = ["min", "rel_energy"]
trials_per_label = 20
folds = 5
svm_parameters = {"C": [1,2], "tol": [0.1,0.01,0.0001]}
rf_parameters = {"n_estimators": [10,100,200], "min_samples_split": [2,5,10]}
knn_parameters = {"n_neighbors": [1,5,10], "p": [1,2]}
ab_parameters = {"n_estimators": [5,10,50]}
mlp_parameters = {"net_specifications": [[20],[10,20]], "batch_normalization": [True, False], "dropout": [True, False], "l2_regularization": [True, False]}
metrics = ["avg_acc", "micro_recall"]
#---------------------------------------

final_data = {}
for level in decomposition_levels:
    final_data[level] = {}
    for time in time_windows:
        final_data[level][time] = {}
        for feature in features:
            final_data[level][time][feature] = {}

            
for i in labels:
    data = create_eeg_matrix(f"eeg_dataset/{i}", True, recording_length, False)
    windows = time_windowing(data, frequency, time_windows, "random", channels=channels, verbose=False)
    for level in decomposition_levels:
        for j in range(len(time_windows)):
            dwt_data = []
            for trial in windows[j]:
                trial_dwt = []
                for channel in trial:
                    approx_coeffs, detail_coeffs = DWT(channel, level, mother_wavelet)
                    trial_dwt.append({"A": approx_coeffs, "D": detail_coeffs})
                dwt_data.append(trial_dwt)
            feature_matrices = {}
            for feature in features:
                feature_matrices[feature] = []
            for trial in dwt_data:
                feature_vector = {}
                for feature in features:
                    feature_vector[feature] = []
                for k in range(len(channels)):
                    channel = trial[k]
                    signals = []
                    for coeff in detail_coefficients[level]:
                        signals.append(channel["D"][coeff-1])
                    for coeff in approximation_coefficients[level]:
                        signals.append(channel["A"][coeff-1])
                    features_vectors = extract_wavelet_features(signals, features)
                    for l in range(len(features)):
                        feature_vector[features[l]].extend(features_vectors[l])
                for feature in features:
                    feature_matrices[feature].append(feature_vector[feature])
            for feature in features:
                final_data[level][time_windows[j]][feature][i] = np.array(feature_matrices[feature])

best_level = None
best_time = None
best_feature = None
best_clf = None
best_params = None
best_metrics = None
max_performance = 0

for level in decomposition_levels:
    for time in time_windows:
        for feature in features:       
            optimization_data, experiment_data = split_data(final_data[level][time][feature], trials_per_label, 0.2, labels)
            
            optimization_folds = create_folds(optimization_data, labels, 4, number_of_folds=folds)
            svm, best_params_svm = optimize_SVM(optimization_folds, svm_parameters)
            rf, best_params_rf = optimize_RF(optimization_folds, rf_parameters)
            knn, best_params_knn = optimize_KNN(optimization_folds, knn_parameters)
            ab_parameters["base_estimator"] = [svm, rf]
            ab, best_params_ab = optimize_AB(optimization_folds, ab_parameters)
            mlp, best_params_mlp = optimize_MLP(optimization_folds, mlp_parameters, (level+1)*len(channels), len(labels), verbose=True)
            
            experiment_folds = create_folds(experiment_data, labels, 16, number_of_folds=folds)
            clfs = [svm,rf,knn,ab,mlp]
            parameters = [best_params_svm,best_params_rf,best_params_knn,best_params_ab,best_params_mlp]
            performance_metrics, confusion_matrices = k_fold_cross_validation(experiment_folds, clfs, len(labels), metrics, "multi", epochs=best_params_mlp[-1])
            
            current_max_performance = 0
            index_max_performance = -1
            for i in range(len(performance_metrics)):
                performance = np.sum(performance_metrics[i])
                if performance > max_performance:
                    current_max_performance = performance
                    index_max_performance = i
            
            if current_max_performance > max_performance:
                best_level = level
                best_time = time
                best_feature = feature
                best_clf = clfs[index_max_performance]
                best_params = parameters[index_max_performance]
                best_metrics = performance_metrics[index_max_performance]
                max_performance = current_max_performance

clf_name = type(best_clf).__name__ if type(best_clf).__name__ != 'Sequential' else 'MLP'

print("\n\n******************* SUMMARY *******************")
print(f"Best decomposition level: {best_level}\nBest time: {best_time}\nBest feature: {best_feature}\nBest classifier: {clf_name}\nBest params: {best_params}")
for i in range(len(metrics)):
    print(f"{metrics[i]}: {best_metrics[i]}")
