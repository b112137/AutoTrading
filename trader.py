#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from numpy import concatenate
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import SGD

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    
    data_all = read_csv(args.training, encoding='utf-8', header=None)
    values = data_all.values
    values = values.astype('float32')
    reframed = series_to_supervised(values, 10, 1)
    reframed.drop(reframed.columns[   [41,42,43]   ]   , axis=1, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(reframed)
    values = scaled
    
    train = values
    train_X, train_y = train[:, :-1], train[:,-1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    print(train_X.shape, train_y.shape)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu',input_shape=(train_X.shape[1], train_X.shape[2]),  return_sequences = True))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(LSTM(units = 50))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=1000, batch_size=32,  verbose=2, shuffle=False)
    
    testing_data = read_csv(args.testing, encoding='utf-8', header=None)
    data_temp = pd.concat( [data_all, testing_data] )
    
    values_temp = data_temp.values
    values_temp = values_temp.astype('float32')
    reframed_temp = series_to_supervised(values_temp, 10, 1)
    reframed_temp.drop(reframed_temp.columns[   [41,42,43]   ]   , axis=1, inplace=True)

    scaled_temp = scaler.transform(reframed_temp)
    values_temp = scaled_temp
    
    test_all = values_temp[-len(testing_data):]
    
    output_X, output_y = test_all[:, :-1], test_all[:, -1]
    output_X = output_X.reshape((output_X.shape[0], 1, output_X.shape[1]))
    yhat = model.predict(output_X)
    output_X = output_X.reshape((output_X.shape[0], output_X.shape[2]))

    inv_yhat = concatenate(( output_X[:, 0:], yhat ), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-1]

    output_y = output_y.reshape((output_y.shape[0], 1))
    inv_y = concatenate(( output_X[:, 0:], output_y ), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]

    
    
    ### action
    have = 0
    action = 0
    action_list = []
    state = []
    predict_state = 0
    predict_state_list = []
    last_open = reframed_temp[-len(testing_data)-1 :]["var1(t)"].iloc[0]

    for i in range(len(inv_y)-1):
        today = inv_y[i]
        if today > last_open:
            state.append(1)
        else:
            state.append(-1)
        
        if inv_yhat[i+1] > inv_yhat[i]:
            predict_state = 1
        else:
            predict_state = -1
        
        predict_state_list.append(predict_state)
        
        if i == 0:
            action = 1
            
            if action == 1:
                have = 1
            elif action == 0:
                have = 0
            elif action == -1:
                have = -1
        else:
            if state[i-1] == 1 and state[i] == 1:
                if have == 1:
                    if predict_state == 1:
                        action = -1
                    else:
                        action = -1
                elif have == 0:
                    if predict_state == 1:
                        action = -1
                    else:
                        action = 0
                elif have == -1:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 0
                    
            elif state[i-1] == 1 and state[i] == -1:
                if have == 1:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 0
                elif have == 0:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 1
                elif have == -1:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 1
                    
            elif state[i-1] == -1 and state[i] == 1:
                if have == 1:
                    if predict_state == 1:
                        action = -1
                    else:
                        action = 0
                elif have == 0:
                    if predict_state == 1:
                        action = 1
                    else:
                        action = -1
                elif have == -1:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 0
                    
            elif state[i-1] == -1 and state[i] == -1:
                if have == 1:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 0
                elif have == 0:
                    if predict_state == 1:
                        action = 0
                    else:
                        action = 0
                elif have == -1:
                    if predict_state == 1:
                        action = 1
                    else:
                        action = 1
        
            if have == 0:
                have = action
            elif have == 1:
                if action == 0:
                    have = 1
                elif action == 1:
                    have = 1
                    print("illegal")
                elif action == -1:
                    have = 0
            elif have == -1:
                if action == 0:
                    have = -1
                elif action == 1:
                    have = 0
                elif action == -1:
                    have = -1
                    print("illegal")
        
        action_list.append(action)
        last_open = today
    
        with open(args.output, 'w') as f:
            for ac in action_list:
                f.writelines(str(ac)+'\n')