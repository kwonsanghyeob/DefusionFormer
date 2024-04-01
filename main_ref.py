import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from models.model import Informer
from data_provider.data_loader import *



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    rawdata = pd.read_csv("./Data/SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv", encoding='CP949')
    individual_station = rawdata[['Unnamed: 0', '종합경기장']]
    tr_l = 7747; va_l = 2582; te_l = int(len(rawdata)-(tr_l+va_l));

    iw = 18;
    ow = 1;
    train_data = individual_station[:tr_l]

    train_dataset = Dataset_ETT_hour(train_data, sequence_length=iw, predict_length=ow, stride=1)


    val_dataset = windowDataset(val_data, sequence_length=iw, predict_length=ow, stride=1)
    test_dataset = windowDataset(test_data, sequence_length=iw, predict_length=ow, stride=1)