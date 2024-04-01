from torch.utils.data import Dataset
import numpy as np
import datetime
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class windowDataset(Dataset):
    def __init__(self, data, sequence_length=80, predict_length=20, stride=5):
        window_length = sequence_length + predict_length

        x = []
        y = []
        for i in range(len(data) - window_length + stride):
            window = data[i:window_length + i, :]
            x.append(window[:-1, :])
            y.append(window[-predict_length:, [0]])

        self.x = np.array(x, dtype='float')
        self.y = np.array(y, dtype='float')

        # self.dec = torch.zeros([self.y.shape[0], self.predict_length, self.y.shape[-1]]).float()

        self.len = len(self.x)
        # 차원수 확인하고 싶으면
        # print(f'x 훈련 데이터 : {x_train.shape}, x 테스트 데이터 : {x_test.shape}, x 훈련 데이터 : {x_val.shape}, x 테스트 데이터 : {x_val.shape}, y 훈련 데이터 : {y_train.shape}, y 테스트 데이터 : {y_test.shape}')

    def __getitem__(self, i):
        return self.x[i], self.y[i,:-1], self.y[i, :1]

    def __len__(self):
        return self.len

def data_preprocessing(data, tr_l = 7747, va_l = 2582, te_l= 50, scaler = 'StandardScaler'):
    '''
    :param data: 입력데이터
    :param window_length: 시퀀스 길이
    :return: X, Y
    '''
    #입력 특징
    if scaler == StandardScaler:
        SCALER = StandardScaler()
    else:
        SCALER = MinMaxScaler()

    new_colum = dict()
    new_colum['Data'] = np.array(data)[:,1]
    new_colum['time_index'] = np.array(pd.to_datetime(data['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S').dt.hour)
    new_colum['Weekday'] = np.array(pd.to_datetime(data['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S').dt.dayofweek)
    new_colum['check_weekend'] = np.array(pd.to_datetime(data['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S').dt.dayofweek>4)

    new_data = pd.DataFrame(new_colum)
    new_data = np.array(pd.get_dummies(new_data, columns=['time_index', 'Weekday', 'check_weekend'], dtype=int))

    train = tr_l
    val = va_l
    test = te_l

    M_data_train = SCALER.fit_transform(new_data[:train, [0]])
    M_data_val = SCALER.fit_transform(new_data[train:(train+val), [0]])
    M_data_test = SCALER.transform(new_data[-test:, [0]])

    EV_charging = np.concatenate((M_data_train, M_data_val,M_data_test))
    EV_charging_new = np.concatenate((EV_charging, new_data[:, 1:]), axis=1)
    return EV_charging_new, SCALER #x_data, y_data, Min_Max, data[-test:, [0]]

if __name__ == '__main__':
    import torch
    import pandas as pd

    device = torch.device("cuda")

    ##데이터 불러오기 및 입력데이터 정의
    rawdata = pd.read_csv("../Data/SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv", encoding='CP949')
    individual_station = rawdata[['Unnamed: 0', '종합경기장']]
    tr_l = 7747; va_l = 2582; te_l = int(len(rawdata)-(tr_l+va_l));
    P_data, Minmax = data_preprocessing(individual_station, tr_l, va_l, te_l)

    train = P_data[:7747]


    iw = 24 * 14
    ow = 1

    train_dataset = windowDataset(train, sequence_length=iw, predict_length=ow, stride=1)
