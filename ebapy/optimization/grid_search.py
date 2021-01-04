from numpy import sum, diag, mean, array, argmax
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import gc


def optimize_SVM(data, parameters, verbose=True):
    C = [1]
    kernel = ["rbf"]
    tol = [0.001]
    gamma = ["scale"]
    
    for parameter in parameters.keys():
        if parameter == "C":
            C = parameters[parameter]
        elif parameter == "kernel":
            kernel = parameters[parameter]
        elif parameter == "tol":
            tol = parameters[parameter]
        elif parameter == "gamma":
            gamm = parameters[parameter]
    
    if verbose: print(f"*********** Starting SVM optimization ***********\nC: {C}\nkernel: {kernel}\ntol: {tol}\ngamma: {gamma}\n")

    acc = []
    parameters = []
    for C_i in C:
        for kernel_i in kernel:
            for tol_i in tol:
                for gamma_i in gamma:
                    accuracies = []
                    for fold in data:
                        X_train = fold["train"]["data"]
                        y_train = fold["train"]["labels"]
                        X_test = fold["test"]["data"]
                        y_test = fold["test"]["labels"]
                        
                        clf = SVC(C=C_i, kernel=kernel_i, tol=tol_i, gamma=gamma_i, probability=True, random_state=2)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test) 
                        cm = confusion_matrix(y_test,y_pred)
                        accuracies.append((sum(diag(cm))/sum(cm))*100)
                    mean_accuracy = mean(accuracies)
                    acc.append(mean_accuracy)
                    parameters.append([C_i, kernel_i, tol_i, gamma_i])
                    if verbose: print(f"C: {C_i}, kernel: {kernel_i}, tol: {tol_i}, gamma: {gamma_i} - Acurracy {mean_accuracy} - DONE")
                    del accuracies, X_train, X_test, y_train, y_test, clf, y_pred, cm, mean_accuracy
                    gc.collect()
    
    best_acc = max(acc)
    best_params = parameters[acc.index(best_acc)]
    if verbose: print(f"\nOptimization finished successfully:\nBest accuracy: {best_acc}\nBest parameters: C: {best_params[0]}, kernel: {best_params[1]}, tol: {best_params[2]}, gamma: {best_params[3]}\n{'*'*49}")
 
    return SVC(C=best_params[0], kernel=best_params[1], tol=best_params[2], gamma=best_params[3], probability=True, random_state=2), best_params


def optimize_RF(data, parameters, verbose=True):
    n_estimators = [100] 
    criterion = ['gini'] 
    min_samples_split = [2] 
    
    for parameter in parameters.keys():
        if parameter == "n_estimators":
            n_estimators = parameters[parameter]
        elif parameter == "criterion":
            criterion = parameters[parameter]
        elif parameter == "min_samples_split":
            min_samples_split = parameters[parameter]
        
    if verbose: print(f"*********** Starting RF optimization ***********\nn_estimators: {n_estimators}\ncriterion: {criterion}\nmin_samples_split: {min_samples_split}\n")
 
    acc = [] 
    parameters = []
    for n_estimators_i in n_estimators:
        for criterion_i in criterion:
            for min_samples_split_i in min_samples_split:
                accuracies = []
                for fold in data:
                    X_train = fold["train"]["data"]
                    y_train = fold["train"]["labels"]
                    X_test = fold["test"]["data"]
                    y_test = fold["test"]["labels"]
                        
                    clf = RandomForestClassifier(n_estimators=n_estimators_i, criterion=criterion_i, min_samples_split=min_samples_split_i, random_state=2)
                    clf.fit(X_train,y_train)
                    y_pred = clf.predict(X_test) 
                    cm = confusion_matrix(y_test,y_pred)
                    accuracies.append((sum(diag(cm))/sum(cm))*100)
                mean_accuracy = mean(accuracies)
                acc.append(mean_accuracy)
                parameters.append([n_estimators_i, criterion_i, min_samples_split_i])
                if verbose: print(f"n_estimators: {n_estimators_i}, criterion: {criterion_i}, min_samples_split: {min_samples_split_i} - Acurracy {mean_accuracy} - DONE")
                del accuracies, X_train, X_test, y_train, y_test, clf, y_pred, cm, mean_accuracy
                gc.collect()

    best_acc = max(acc)
    best_params = parameters[acc.index(best_acc)]
    if verbose: print(f"\nOptimization finished successfully:\nBest accuracy: {best_acc}\nBest parameters: n_estimators: {best_params[0]}, criterion: {best_params[1]}, min_samples_split: {best_params[2]}\n{'*'*49}")
 
    return RandomForestClassifier(n_estimators=best_params[0], criterion=best_params[1], min_samples_split=best_params[2], random_state=2), best_params
        

def optimize_KNN(data, parameters, verbose=True):
    n_neighbors = [5] 
    leaf_size = [30] 
    p = [2] 
    
    for parameter in parameters.keys():
        if parameter == "n_neighbors":
            n_neighbors = parameters[parameter]
        elif parameter == "leaf_size":
            leaf_size = parameters[parameter]
        elif parameter == "p":
            p = parameters[parameter]
    
    if verbose: print(f"*********** Starting KNN optimization ***********\nn_neighbors: {n_neighbors}\nleaf_size: {leaf_size}\np: {p}\n")
 
    acc = [] 
    parameters = []
    for n_neighbors_i in n_neighbors:
        for leaf_size_i in leaf_size:
            for p_i in p:
                accuracies = []
                for fold in data:
                    X_train = fold["train"]["data"]
                    y_train = fold["train"]["labels"]
                    X_test = fold["test"]["data"]
                    y_test = fold["test"]["labels"]
                    
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors_i, leaf_size=leaf_size_i, p=p_i, n_jobs=-1)
                    clf.fit(X_train,y_train)
                    y_pred = clf.predict(X_test) 
                    cm = confusion_matrix(y_test,y_pred)
                    accuracies.append((sum(diag(cm))/sum(cm))*100)
                mean_accuracy = mean(accuracies)
                acc.append(mean_accuracy)
                parameters.append([n_neighbors_i, leaf_size_i, p_i])
                if verbose: print(f"n_neighbors: {n_neighbors_i}, leaf_size: {leaf_size_i}, p: {p_i} - Acurracy {mean_accuracy} - DONE")
                del accuracies, X_train, X_test, y_train, y_test, clf, y_pred, cm, mean_accuracy
                gc.collect()
    
    best_acc = max(acc)
    best_params = parameters[acc.index(best_acc)]
    if verbose: print(f"\nOptimization finished successfully:\nBest accuracy: {best_acc}\nBest parameters: n_neighbors: {best_params[0]}, leaf_size: {best_params[1]}, p: {best_params[2]}\n{'*'*49}")
 
    return KNeighborsClassifier(n_neighbors=best_params[0], leaf_size=best_params[1], p=best_params[2], n_jobs=-1), best_params
                
                
def optimize_AB(data, parameters, verbose=True):
    base_estimator = [None] 
    n_estimators = [50] 
    learning_rate = [1] 
    algorithm = ['SAMME.R']
    
    for parameter in parameters.keys():
        if parameter == "base_estimator":
            base_estimator = parameters[parameter]
        elif parameter == "n_estimators":
            n_estimators = parameters[parameter]
        elif parameter == "learning_rate":
            learning_rate = parameters[parameter]
        elif parameter == "algorithm":
            algorithm = parameters[parameter]
    
    if verbose: print(f"*********** Starting AB optimization ***********\nbase_estimator: {base_estimator}\nn_estimators: {n_estimators}\nlearning_rate: {learning_rate}\nalgorithm: {algorithm}\n")
    
    acc = []
    parameters = []
    for base_estimator_i in base_estimator:
        for n_estimators_i in n_estimators:
            for learning_rate_i in learning_rate:
                for algorithm_i in algorithm:
                    accuracies = []
                    for fold in data:
                        X_train = fold["train"]["data"]
                        y_train = fold["train"]["labels"]
                        X_test = fold["test"]["data"]
                        y_test = fold["test"]["labels"]
                    
                        clf = AdaBoostClassifier(base_estimator=base_estimator_i, n_estimators=n_estimators_i, learning_rate=learning_rate_i, algorithm=algorithm_i, random_state=2)
                        clf.fit(X_train,y_train)
                        y_pred = clf.predict(X_test) 
                        cm = confusion_matrix(y_test,y_pred)
                        accuracies.append((sum(diag(cm))/sum(cm))*100)
                    mean_accuracy = mean(accuracies)
                    acc.append(mean_accuracy)
                    parameters.append([base_estimator_i, n_estimators_i, learning_rate_i, algorithm_i])
                    if verbose: print(f"base_estimator: {type(base_estimator_i).__name__}, n_estimators: {n_estimators_i}, learning_rate: {learning_rate_i}, algorithm: {algorithm_i} - Acurracy {mean_accuracy} - DONE")
                    del accuracies, X_train, X_test, y_train, y_test, clf, y_pred, cm, mean_accuracy
                    gc.collect()
    
    best_acc = max(acc)
    best_params = parameters[acc.index(best_acc)]
    if verbose: print(f"\nOptimization finished successfully:\nBest accuracy: {best_acc}\nBest parameters: base_estimator: {type(best_params[0]).__name__}, n_estimators: {best_params[1]}, learning_rate: {best_params[2]}, algorithm: {best_params[3]}\n{'*'*49}")
 
    return AdaBoostClassifier(base_estimator=best_params[0], n_estimators=best_params[1], learning_rate=best_params[2], algorithm=best_params[3], random_state=2), best_params
        

def change_prediction_vector(y_pred):
    y_pred_final = []
    for pred in y_pred:
        y_pred_final.append(np.argmax(pred)+1)
    return y_pred_final


def optimize_MLP(data, parameters, input_dim, number_of_labels, verbose=True):  
    net_specifications = [[100]] 
    learning_rate = [1e-3] 
    batch_normalization = [False] 
    dropout = [True] 
    dropout_percentage = [0.2] 
    l2_regularization = [True] 
    l2_regularization_values = [0.01] 
    epochs = [10] 
    
    for parameter in parameters.keys():
        if parameter == "net_specifications":
            net_specifications = parameters[parameter]
        elif parameter == "learning_rate":
            learning_rate = parameters[parameter]
        elif parameter == "batch_normalization":
            batch_normalization = parameters[parameter]
        elif parameter == "dropout":
            dropout = parameters[parameter]
        elif parameter == "dropout_percentage":
            dropout_percentage = parameters[parameter]
        elif parameter == "l2_regularization":
            l2_regularization = parameters[parameter]
        elif parameter == "l2_regularization_values":
            l2_regularization_values = parameters[parameter]
        elif parameter == "epochs":
            epochs = parameters[parameter]
    
    if verbose: print(f"*********** Starting MLP optimization ***********\nnet_specifications: {net_specifications}\nlearning_rate: {learning_rate}\nbatch_normalization: {batch_normalization}\nbatch_normalization: {batch_normalization}\ndropout: {dropout}\ndropout_percentage: {dropout_percentage}\nl2_regularization: {l2_regularization}\nl2_regularization_values: {l2_regularization_values}\nepochs: {epochs}\n")
    
    acc = [] 
    parameters = []
    for net_specifications_i in net_specifications:
        for learning_rate_i in learning_rate:
            for batch_normalization_i in batch_normalization:
                for dropout_i in dropout:
                    for l2_regularization_i in l2_regularization:
                        for epochs_i in epochs:
                            
                            if dropout_i and l2_regularization_i:
                                for dropout_percentage_i in dropout_percentage:                          
                                    for l2_regularization_values_i in l2_regularization_values:
                                        model = Sequential()
                                        for layer in range(len(net_specifications_i)):
                                            if layer == 0:
                                                model.add(Dense(units=net_specifications_i[layer], input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(l2_regularization_values_i)))
                                            else:
                                                model.add(Dense(units=net_specifications_i[layer], kernel_regularizer=l2(l2_regularization_values_i)))
                                            if batch_normalization_i:
                                                model.add(BatchNormalization())
                                            model.add(Dropout(dropout_percentage_i))
                                            model.add(Activation('relu'))  
                                        model.add(Dense(units=number_of_labels, activation='softmax'))
                                        adam = Adam(learning_rate=learning_rate_i)
                                        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                        
                                        accuracies = []
                                        for fold in data:
                                            X_train = fold["train"]["data"]
                                            y_train = fold["train"]["labels"]
                                            X_test = fold["test"]["data"]
                                            y_test = fold["test"]["labels"]
											
                                            enc = OneHotEncoder(categories="auto")
                                            y_train = array(y_train).reshape(len(y_train), 1)
                                            y_train = enc.fit_transform(y_train).toarray()
                                            model.fit(X_train, y_train, batch_size=100, epochs=epochs_i, verbose=0)
                                            y_pred = model.predict(X_test) 
                                            y_pred = array(list(map(argmax, y_pred)))+1
                                            cm = confusion_matrix(y_test,y_pred)
                                            accuracies.append((sum(diag(cm))/sum(cm))*100)
                                        mean_accuracy = mean(accuracies)
                                        acc.append(mean_accuracy)
                                        parameters.append([net_specifications_i, learning_rate_i, batch_normalization_i, dropout_i, dropout_percentage_i, l2_regularization_i, l2_regularization_values_i, epochs_i])
                                        if verbose: print(f"net_specifications: {net_specifications_i}, learning_rate: {learning_rate_i}, batch_normalization: {batch_normalization_i}, dropout: {dropout_i}, dropout_percentage: {dropout_percentage_i}, l2_regularization: {l2_regularization_i}, l2_regularization_values: {l2_regularization_values_i}, epochs: {epochs_i} - Acurracy {mean_accuracy} - DONE")
                                        del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred, cm, mean_accuracy
                                        K.clear_session()
                                        gc.collect()
                                          
                            elif dropout_i:
                                for dropout_percentage_i in dropout_percentage:  
                                    model = Sequential()
                                    for layer in range(len(net_specifications_i)):     
                                        if layer == 0:
                                            model.add(Dense(units=net_specifications_i[layer], input_dim=input_dim, kernel_initializer='uniform'))
                                        else:
                                            model.add(Dense(units=net_specifications_i[layer]))
                                        if batch_normalization_i:
                                            model.add(BatchNormalization())
                                        model.add(Dropout(dropout_percentage_i))
                                        model.add(Activation('relu'))  
                                    model.add(Dense(units=number_of_labels, activation='softmax'))
                                    adam = Adam(learning_rate=learning_rate_i)
                                    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                        
                                    accuracies = []
                                    for fold in data:
                                        X_train = fold["train"]["data"]
                                        y_train = fold["train"]["labels"]
                                        X_test = fold["test"]["data"]
                                        y_test = fold["test"]["labels"]
                                        
                                        enc = OneHotEncoder(categories="auto")
                                        y_train = array(y_train).reshape(len(y_train), 1)
                                        y_train = enc.fit_transform(y_train).toarray()
                                        model.fit(X_train, y_train, batch_size=100, epochs=epochs_i, verbose=0)
                                        y_pred = model.predict(X_test) 
                                        y_pred = array(list(map(argmax, y_pred)))+1
                                        cm = confusion_matrix(y_test,y_pred)
                                        accuracies.append((sum(diag(cm))/sum(cm))*100)
                                    mean_accuracy = mean(accuracies)
                                    acc.append(mean_accuracy)
                                    parameters.append([net_specifications_i, learning_rate_i, batch_normalization_i, dropout_i, dropout_percentage_i, l2_regularization_i, None, epochs_i])
                                    if verbose: print(f"net_specifications: {net_specifications_i}, learning_rate: {learning_rate_i}, batch_normalization: {batch_normalization_i}, dropout: {dropout_i}, dropout_percentage: {dropout_percentage_i}, l2_regularization: {l2_regularization_i}, epochs: {epochs_i} - Acurracy {mean_accuracy} - DONE")
                                    del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred, cm, mean_accuracy
                                    K.clear_session()
                                    gc.collect()
                                    
                                             
                            elif l2_regularization_i:                         
                                for l2_regularization_values_i in l2_regularization_values:
                                    model = Sequential()
                                    for layer in range(len(net_specifications_i)):
                                        if layer == 0:
                                            model.add(Dense(units=net_specifications_i[layer], input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(l2_regularization_values_i)))
                                        else:
                                            model.add(Dense(units=net_specifications_i[layer], kernel_regularizer=l2(l2_regularization_values_i)))
                                        if batch_normalization_i:
                                            model.add(BatchNormalization())
                                        model.add(Activation('relu'))  
                                    model.add(Dense(units=number_of_labels, activation='softmax'))
                                    adam = Adam(learning_rate=learning_rate_i)
                                    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                    
                                    accuracies = []
                                    for fold in data:
                                        X_train = fold["train"]["data"]
                                        y_train = fold["train"]["labels"]
                                        X_test = fold["test"]["data"]
                                        y_test = fold["test"]["labels"]
										
                                        enc = OneHotEncoder(categories="auto")
                                        y_train = array(y_train).reshape(len(y_train), 1)
                                        y_train = enc.fit_transform(y_train).toarray()
                                        model.fit(X_train, y_train, batch_size=100, epochs=epochs_i, verbose=0)
                                        y_pred = model.predict(X_test) 
                                        y_pred = array(list(map(argmax, y_pred)))+1
                                        cm = confusion_matrix(y_test,y_pred)
                                        accuracies.append((sum(diag(cm))/sum(cm))*100)
                                    mean_accuracy = mean(accuracies)
                                    acc.append(mean_accuracy)
                                    parameters.append([net_specifications_i, learning_rate_i, batch_normalization_i, dropout_i, None, l2_regularization_i, l2_regularization_values_i, epochs_i])
                                    if verbose: print(f"net_specifications: {net_specifications_i}, learning_rate: {learning_rate_i}, batch_normalization: {batch_normalization_i}, dropout: {dropout_i}, l2_regularization: {l2_regularization_i}, l2_regularization_values: {l2_regularization_values_i}, epochs: {epochs_i} - Acurracy {mean_accuracy} - DONE")
                                    del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred, cm, mean_accuracy
                                    K.clear_session()
                                    gc.collect()
                                                           
                            else:
                                model = Sequential()
                                for layer in range(len(net_specifications_i)):
                                    if layer == 0:
                                        model.add(Dense(units=net_specifications_i[layer], input_dim=input_dim, kernel_initializer='uniform'))
                                    else:
                                        model.add(Dense(units=net_specifications_i[layer]))
                                    if batch_normalization_i:
                                        model.add(BatchNormalization())
                                    model.add(Activation('relu'))  
                                model.add(Dense(units=number_of_labels, activation='softmax'))
                                adam = Adam(learning_rate=learning_rate_i)
                                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                
                                accuracies = []
                                for fold in data:
                                    X_train = fold["train"]["data"]
                                    y_train = fold["train"]["labels"]
                                    X_test = fold["test"]["data"]
                                    y_test = fold["test"]["labels"]
                                    
                                    enc = OneHotEncoder(categories="auto")
                                    y_train = array(y_train).reshape(len(y_train), 1)
                                    y_train = enc.fit_transform(y_train).toarray()
                                    model.fit(X_train, y_train, batch_size=100, epochs=epochs_i, verbose=0)
                                    y_pred = model.predict(X_test) 
                                    y_pred = array(list(map(argmax, y_pred)))+1
                                    cm = confusion_matrix(y_test,y_pred)
                                    accuracies.append((sum(diag(cm))/sum(cm))*100)
                                mean_accuracy = mean(accuracies)
                                acc.append(mean_accuracy)
                                parameters.append([net_specifications_i, learning_rate_i, batch_normalization_i, dropout_i, None, l2_regularization_i, None, epochs_i])
                                if verbose: print(f"net_specifications: {net_specifications_i}, learning_rate: {learning_rate_i}, batch_normalization: {batch_normalization_i}, dropout: {dropout_i}, l2_regularization: {l2_regularization_i}, epochs: {epochs_i} - Acurracy {mean_accuracy} - DONE")
                                del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred, cm, mean_accuracy
                                K.clear_session()
                                gc.collect()
                                                   
    best_acc = max(acc)
    best_params = parameters[acc.index(best_acc)]
    if verbose: print(f"\nOptimization finished successfully:\nBest accuracy: {best_acc}\nBest parameters: net_specifications: {best_params[0]}, learning_rate: {best_params[1]}, batch_normalization: {best_params[2]}, dropout: {best_params[3]}, dropout_percentage: {best_params[4]}, l2_regularization: {best_params[5]}, l2_regularization_values: {best_params[6]}, epochs: {best_params[7]}\n{'*'*49}")

    model = Sequential()
    if best_params[3] and best_params[5]:
        for layer in range(len(best_params[0])):
            if layer == 0:
                model.add(Dense(units=best_params[0][layer], input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(best_params[6])))
            else:
                model.add(Dense(units=best_params[0][layer], kernel_regularizer=l2(best_params[6])))
            if best_params[2]:
                model.add(BatchNormalization())
            model.add(Dropout(best_params[4]))
            model.add(Activation('relu'))  
        model.add(Dense(units=number_of_labels, activation='softmax'))
        adam = Adam(learning_rate=best_params[1])
    elif best_params[3]:
        for layer in range(len(best_params[0])):
            if layer == 0:
                model.add(Dense(units=best_params[0][layer], input_dim=input_dim, kernel_initializer='uniform'))
            else:
                model.add(Dense(units=best_params[0][layer]))
            if best_params[2]:
                model.add(BatchNormalization())
            model.add(Dropout(best_params[4]))
            model.add(Activation('relu'))  
        model.add(Dense(units=number_of_labels, activation='softmax'))
        adam = Adam(learning_rate=best_params[1])
    elif best_params[5]:
        for layer in range(len(best_params[0])):
            if layer == 0:
                model.add(Dense(units=best_params[0][layer], input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(best_params[6])))
            else:
                model.add(Dense(units=best_params[0][layer], kernel_regularizer=l2(best_params[6])))
            if best_params[2]:
                model.add(BatchNormalization())
            model.add(Activation('relu'))  
        model.add(Dense(units=number_of_labels, activation='softmax'))
        adam = Adam(learning_rate=best_params[1])
    else:
        for layer in range(len(best_params[0])):
            if layer == 0:
                model.add(Dense(units=best_params[0][layer], input_dim=input_dim, kernel_initializer='uniform'))
            else:
                model.add(Dense(units=best_params[0][layer]))
            if best_params[2]:
                model.add(BatchNormalization())
            model.add(Activation('relu'))  
        model.add(Dense(units=number_of_labels, activation='softmax'))
        adam = Adam(learning_rate=best_params[1])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model, best_params
        
                                