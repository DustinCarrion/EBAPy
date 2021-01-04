from numpy import array
from numpy.random import choice, shuffle
from sklearn.preprocessing import MinMaxScaler
    
def split_data(data, trials_per_label, percentage_for_optimization, labels):
    optimization_indexes_per_class = []
    for _ in labels: 
        optimization_indexes_per_class.append(choice(trials_per_label, int(percentage_for_optimization*trials_per_label), replace=False))
    
    experiment_indexes_per_class = []
    for i in range(len(labels)):
        indexes = []
        for j in range(trials_per_label):
            if j not in optimization_indexes_per_class[i]:
                indexes.append(j)
        experiment_indexes_per_class.append(indexes)
    
    optimization_data = {}
    experiment_data = {}
    
    for i in range(len(labels)):
        optimization_data[labels[i]] = []
        experiment_data[labels[i]] = []
        
        label_data = data[labels[i]]                          
        for trial in optimization_indexes_per_class[i]:
            optimization_data[labels[i]].append(label_data[trial])
        optimization_data[labels[i]] = array(optimization_data[labels[i]])
        
        for trial in experiment_indexes_per_class[i]:
            experiment_data[labels[i]].append(label_data[trial])
        experiment_data[labels[i]] = array(experiment_data[labels[i]])
        
    return optimization_data, experiment_data


def train_test_split(data, labels, train_indexes, test_indexes):    
    X_train_ordered = []
    X_test_ordered = []
    y_train_ordered = []
    y_test_ordered = []
    for i in range(len(labels)):
        X_train_aux = [] 
        y_train_aux = []
        for trial in train_indexes[i]:
            X_train_aux.append(data[labels[i]][trial])
            y_train_aux.append(labels[i])
        
        X_test_aux = [] 
        y_test_aux = []
        for trial in test_indexes[i]:
            X_test_aux.append(data[labels[i]][trial])
            y_test_aux.append(labels[i])
            
        X_train_ordered.extend(X_train_aux)
        X_test_ordered.extend(X_test_aux)
        y_train_ordered.extend(y_train_aux)
        y_test_ordered.extend(y_test_aux)
    
    train_shuffle = choice(len(X_train_ordered),len(X_train_ordered),replace=False)
    X_train = []
    y_train = []
    for i in train_shuffle:
        X_train.append(X_train_ordered[i])
        y_train.append(y_train_ordered[i])
    X_train = array(X_train)
    
    test_shuffle = choice(len(X_test_ordered),len(X_test_ordered),replace=False)
    X_test = []
    y_test = []
    for i in test_shuffle:
        X_test.append(X_test_ordered[i])
        y_test.append(y_test_ordered[i])
    X_test = array(X_test)
    
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    train = {'data':X_train, 'labels': y_train}
    test = {'data':X_test, 'labels': y_test}
    return train, test


def create_folds(data, labels, trials_per_label, train_size=0.75, number_of_folds=10):
    folds = []
    for fold in range(number_of_folds):
        train_indexes = []
        for _ in labels: 
            train_indexes.append(choice(trials_per_label, int(train_size*trials_per_label), replace=False))
        
        test_indexes = []
        for i in range(len(labels)):
            indexes = []
            for trial in range(trials_per_label):
                if trial not in train_indexes[i]:
                    indexes.append(trial)
            shuffle(indexes)
            test_indexes.append(indexes)
        
        train_data, test_data = train_test_split(data, labels, train_indexes, test_indexes)
        folds.append({'train': train_data, 'test': test_data})
    return folds
    