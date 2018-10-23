import random
import numpy as np
import pandas as pd
import csv


#separate a proportional training set
#data is a list of dictionaries
#y the name of the response or class
#size is the proportion reserved to the training set (default = 0.2)

def train_test(data,y,size=0.2):

    ''' returns two dataframes for training and testing
    -- data is a list of dictionaries
    -- y is the response or class (used to asssure the train and test data
    frames have equal proportions of response)
    Currently set for bianary responses.
    '''
    y_list = list()
    for i in data:
        y_value = (i[y])
        y_list.append(y_value)
        y_set = list(set(y_list))

#y_1 and y_2 are the count of each class
    y_1 = y_list.count(y_set[-1])
    y_2 = y_list.count(y_set[0])
    y_total = y_1 + y_2

    p_1 = y_1 / y_total
  
#the list of dictionaries are shuffled (in case there were ordered)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

#calcuate size of sets to return - use ceiling and abolute value to prevent fractions
    #of training and test set sizes
    train_size = np.ceil(y_total * size).astype(int)

#size of training set responses
    train_y_1 = np.ceil(train_size * p_1).astype(int)
    train_y_2 = np.abs(train_size - train_y_1).astype(int)

#slice dataframe by set sizes 
    for i in data:
        if (i[y] == y_set[-1]):
           list_train_y_1 = data[:train_y_1]
           list_test_y_1 = data[train_y_1:y_1]
        else:
            list_train_y_2 = data[:train_y_2]
            list_test_y_2 = data[train_y_2:y_2]

#contact the training and test sets, then shuffle for no paticular reason
    train = list_train_y_1 + list_train_y_2
    test = list_test_y_1 + list_test_y_2

    random.shuffle(train)
    random.shuffle(test)

    for i in train:
        df_train = pd.DataFrame(train)
        df_test = pd.DataFrame(test)

    df_train = df_train[['diagnosis', 'mean_radius', 'mean_texture', 
                           'mean_perimeter', 'mean_area', 'mean_smoothness',
                           'mean_compactness', 'mean_concavity', 'mean_concave_points',
                           'mean_symmetry' ,'mean_fractal_dimension']]
    df_test = df_test[['diagnosis', 'mean_radius', 'mean_texture', 
                           'mean_perimeter', 'mean_area', 'mean_smoothness',
                           'mean_compactness', 'mean_concavity', 'mean_concave_points',
                           'mean_symmetry' ,'mean_fractal_dimension']]

    codes = {'B':0, 'M':1}

    df_train['coded'] = df_train['diagnosis'].map(codes)
    df_test['coded'] = df_test['diagnosis'].map(codes)

    df_train.to_csv('./data/train.csv')
    df_test.to_csv('./data/test.csv')

    return df_train, df_test
    
    
    
