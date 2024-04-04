import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns


if __name__ == '__main__':
    directory = glob.glob('./results_hyper_Label/*')
    data_result = []
    for dit in directory:
        data_result.append(np.load(f'{dit}/metrics.npy')[:2])

    result = np.array(data_result)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', size=16)

    plt.figure(num = 1, figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('MAE')
    plt.plot(result[:,0], linestyle='dashed',marker='o', color = 'red', linewidth = 3)
    plt.grid()


    plt.subplot(1, 2, 2)
    plt.title('MAE')
    plt.plot(result[:, 1], linestyle='dashed',marker='o', color = 'red', linewidth = 3)
    plt.grid()
    plt.tight_layout()

    plt.show()