import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from prophet import Prophet
from Input_package import windowDataset, data_preprocessing
from paper_package import *




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    rawdata = pd.read_csv("./Data/SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv", encoding='CP949')
    individual_station = dict()
    individual_station['ds'] = np.array(rawdata['Unnamed: 0'])
    individual_station['y'] = np.array(rawdata['종합경기장'])
    data = pd.DataFrame(individual_station)
    tr_l = 7747; va_l = 2582; te_l = int(len(rawdata)-(tr_l+va_l));
    test = data[-int(len(rawdata)-(tr_l+va_l)):]['y']

    m = Prophet()
    m.fit(data[:-(tr_l+va_l)])

    future = m.make_future_dataframe(freq='h', periods=1)
    forecast = m.predict(future)
    pred = forecast['yhat']

    print(f'MSE : {MSE(np.array(test), np.array(pred)[1:])}')
    print(f'RSME : {RSME(np.array(test), np.array(pred)[1:])}')
    print(f'MAE : {MAE(np.array(test), np.array(pred)[1:])}')
    print(f'MAPE : {masked_mape_np(np.array(test), np.array(pred)[1:])}')

    result = dict()
    result['real'] = test
    result['predict'] = pred[1:]
    pd.DataFrame(result).to_csv('result/result_Prophet.csv')


    forecast[['ds', 'yhat']].tail()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 5))
    plt.plot(np.array(forecast['yhat'])[-te_l:])
    plt.plot(np.array(individual_station['y'])[-te_l:])
    plt.show()

    print(f'MSE : {MSE(y.reshape(-1, 1), Min_Max.inverse_transform(y_predict.reshape(-1, 1)))}')
    print(f'RSME : {RSME(y.reshape(-1, 1), Min_Max.inverse_transform(y_predict.reshape(-1, 1)))}')
    print(f'MAE : {MAE(y.reshape(-1, 1), Min_Max.inverse_transform(y_predict.reshape(-1, 1)))}')
    print(f'MAPE : {masked_mape_np(y.reshape(-1, 1), Min_Max.inverse_transform(y_predict.reshape(-1, 1)))}')
