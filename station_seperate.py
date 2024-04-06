import pandas as pd
import numpy as np

import os
os.getcwd()
data = pd.read_csv('./Data/SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv', encoding='cp949')
data.columns
data_1 = data[[data.columns[0],'제주특별자치도청']]
data_2 = data[[data.columns[0],'제주시청']]
data_3 = data[[data.columns[0],'탐라도서관']]
data_4 = data[[data.columns[0],'제주특별자치도의회']]
data_5 = data[[data.columns[0],'제스코마트']]
data_6 = data[[data.columns[0],'제주공항']]
data_7 = data[[data.columns[0],'노형제1공영주차장']]
data_8 = data[[data.columns[0],'모로왓제2공영주차장']]

data_1.to_csv('test_1.csv')
data_2.to_csv('test_2.csv')
data_3.to_csv('test_3.csv')
data_4.to_csv('test_4.csv')
data_5.to_csv('test_5.csv')
data_6.to_csv('test_6.csv')
data_7.to_csv('test_7.csv')
data_8.to_csv('test_8.csv')