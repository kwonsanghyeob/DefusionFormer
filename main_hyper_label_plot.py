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

    LL = []
    for i in range(6, 102, 6):
        LL.append(i)

    result = np.array(data_result)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', size=25)

    plt.figure(num = 1, figsize=(10,6))
    # plt.subplot(2,1,1)
    plt.plot(LL,result[:,0], linestyle='dashed',marker='o', color = 'red', linewidth = 3)
    # plt.axvspan(65,67,color = 'grey', alpha = 1)
    plt.annotate('Optimal label length', xy=(25, min(result[:, 0])), xytext =(35,min(result[:, 0])+0.005),arrowprops=dict(facecolor='green', shrink=0.001))
    plt.ylabel('MSE')
    plt.xticks(LL)
    plt.xlabel('label length')
    plt.grid()
    plt.tight_layout()
    plt.show()

    #
    # plt.subplot(2,1, 2)
    # plt.title('MAE')
    # plt.plot(LL,result[:, 1], linestyle='dashed',marker='o', color = 'red', linewidth = 3)
    # # plt.axvspan(65, 67, color='grey', alpha=1)
    # plt.annotate('Optimal label length', xy=(25, min(result[:, 1])+0.001), xytext=(35, min(result[:, 1])), arrowprops=dict(facecolor='green', shrink=0.001))
    # plt.xlabel('Label length')
    # plt.xticks(LL)
    # plt.grid()
    # plt.tight_layout()

    plt.show()