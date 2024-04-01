import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def time_features(dates, freq='h'):
    dates['month'] = dates.date.apply(lambda row:row.month,1)
    dates['day'] = dates.date.apply(lambda row:row.day,1)
    dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
    dates['hour'] = dates.date.apply(lambda row:row.hour,1)
    dates['minute'] = dates.date.apply(lambda row:row.minute,1)
    dates['minute'] = dates.minute.map(lambda x:x//15)
    freq_map = {
        'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
        'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
        't':['month','day','weekday','hour','minute'],
    }
    return dates[freq_map[freq.lower()]].values


class Dataset_Pred(Dataset):
    def __init__(self, dataframe, size=None, scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dataframe = dataframe

        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.dataframe
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        delta = df_raw["date"].iloc[1] - df_raw["date"].iloc[0]
        if delta >= timedelta(hours=1):
            self.freq = 'h'
        else:
            self.freq = 't'

        border1 = 0
        border2 = len(df_raw)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


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