from torch.utils.data import Dataset
import numpy as np



class windowDataset_test(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        # 총 데이터의 개수
        L = y.shape[0]
        # stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        # input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))
        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i, :-1], self.y[i, 1:]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    import torch
    import pandas as pd

    device = torch.device("cuda")

    ##데이터 불러오기 및 입력데이터 정의
    rawdata = pd.read_csv("../서인천IC-부평IC 평균속도.csv", encoding='CP949')

    train = rawdata[:-24 * 7]
    data_train = train["평균속도"].to_numpy()

    iw = 24 * 14
    ow = 24 * 7

    train_dataset = windowDataset_test(data_train, input_window=iw, output_window=ow, stride=1)


