import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_seperate_V1(data, sequence_length = 24, predict_length = 1):
    '''
    :param data: 입력데이터
    :param window_length: 시퀀스 길이
    :return: X, Y
    '''
    #입력 특징
    data['time_index'] = None
    data['check_weekend'] = None
    data['Weekday'] = None
    data['check_weekend'] = None

    for i in range(len(data)):
        data['time_index'].values[i] = datetime.datetime.strptime(data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').time().strftime('%H:%M')

    for i in range(len(data)):
        Week_value= datetime.datetime.strptime(data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').weekday()
        data['Weekday'].values[i] = Week_value
        if int(Week_value) <= 4:
            data['check_weekend'].values[i] = 0
        elif int(Week_value) > 4:
            data['check_weekend'].values[i] = 1

    data = pd.get_dummies(data, columns=['time_index', 'Weekday', 'check_weekend'], dtype=int)
    data = np.array(data.drop(['Unnamed: 0'], axis=1))

    Min_Max = MinMaxScaler()

    train = int(len(data)*0.6)
    val = int(len(data) * 0.2)
    test = int(len(data) - (train+val))

    M_data_train = Min_Max.fit_transform(data[:train, [0]])
    M_data_val = Min_Max.fit_transform(data[train:(train+val), [0]])
    M_data_test = Min_Max.transform(data[-test:, [0]])

    EV_charging = np.concatenate((M_data_train, M_data_val,M_data_test))
    EV_charging_new = np.concatenate((EV_charging, data[:, 1:]), axis=1)

    window_length = sequence_length + predict_length

    x = []
    y = []

    for i in range(len(data)-window_length+1):
        window = EV_charging_new[i:window_length+i,:]
        x.append(window[:-1,:])
        y.append(window[-1,0])

    np_x = np.array(x)
    np_y = np.array(y)

    ###
    x_train = np_x[:train]
    y_train = np_y[:train]

    x_val = np_x[train:(train+val)]
    y_val = np_y[train:(train+val)]

    x_test = np_x[-test:]
    y_test = np_y[-test:]

    ###
    x_data = {
        'train': x_train,
        'val': x_val,
        'test': x_test
    }

    y_data = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }

    #차원수 확인하고 싶으면
    # print(f'x 훈련 데이터 : {x_train.shape}, x 테스트 데이터 : {x_test.shape}, x 훈련 데이터 : {x_val.shape}, x 테스트 데이터 : {x_val.shape}, y 훈련 데이터 : {y_train.shape}, y 테스트 데이터 : {y_test.shape}')

    return x_data, y_data, Min_Max, data[-test:, [0]]

def multi_input_V1(data, long_length = 300, mid_length =200, short_length = 100, dim_3 = True, predict_length = 1):
    '''
    :param data: 입력데이터
    :param window_length: 시퀀스 길이
    :return: X, Y
    '''


    Min_Max = MinMaxScaler()
    data['time_index'] = None
    data['check_weekend'] = None
    for i in range(len(data)):
        data['time_index'].values[i] = datetime.datetime.strptime(data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').time().strftime('%H:%M')

    data['Weekday'] = None
    data['check_weekend'] = None
    for i in range(len(data)):
        Week_value= datetime.datetime.strptime(data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').weekday()
        data['Weekday'].values[i] = Week_value
        if int(Week_value) <= 4:
            data['check_weekend'].values[i] = 0
        elif int(Week_value) > 4:
            data['check_weekend'].values[i] = 1

    Data = pd.get_dummies(data, columns=['time_index', 'Weekday', 'check_weekend'], dtype=int)
    data = np.array(Data.drop(['Unnamed: 0'], axis=1))

    train = int(len(data)*0.6)
    val = int(len(data) * 0.2)
    test = int(len(data) - (train+val))

    M_data_train = Min_Max.fit_transform(data[:train, [0]])
    M_data_val = Min_Max.fit_transform(data[train:(train+val), [0]])
    M_data_test = Min_Max.transform(data[-test:, [0]])

    EV_charging = np.concatenate((M_data_train, M_data_val,M_data_test))
    EV_charging_new = np.concatenate((EV_charging, data[:, 1:]), axis=1)


    window_length = long_length+predict_length
    x_short = []
    x_mid = []
    x_long = []
    y = []

    for i in range(len(EV_charging_new)-window_length+1):
        window = EV_charging_new[i:window_length+i,:]
        x_long.append(window[:-1,:])
        x_mid.append(window[-mid_length-1:-1,:])
        x_short.append(window[-short_length-1:-1,:])
        y.append(window[-1,0])

    np_xlong = np.array(x_long)
    np_xmid = np.array(x_mid)
    np_xshort = np.array(x_short)
    np_y = np.array(y)


    #장기 데이터 나누기
    xlong_train = np_xlong[:train]
    xlong_val = np_xlong[train:(train+val)]
    xlong_test = np_xlong[-test:]

    #중기 데이터 나누기
    xmid_train = np_xmid[:train]
    xmid_val = np_xmid[train:(train+val)]
    xmid_test = np_xmid[-test:]

    #단기 데이터 나누기
    xshort_train = np_xshort[:train]
    xshort_val = np_xlong[train:(train + val)]
    xshort_test = np_xshort[-test:]

    #target
    target_train = np_y[:train]
    target_val = np_y[train:(train + val)]
    target_test = np_y[-test:]

    x_data = {
        'train' : {
            'long' : xlong_train,
            'mid' : xmid_train,
            'short' : xshort_train
        },
        'val': {
            'long': xlong_val,
            'mid': xmid_val,
            'short': xshort_val
        },
        'test' : {
            'long': xlong_test,
            'mid': xmid_test,
            'short': xshort_test
        }
    }

    y_data = {
        'train': {
            'target': target_train
        },
        'val': {
            'target': target_val,
        },
        'test': {
            'target': target_test
        }
    }

    return x_data, y_data, Min_Max, data[-test:, [0]]