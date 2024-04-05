import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns

def heatmap_plot(L, M, S):
    dir_last = []
    for Long in L:
        dir_Coll = []
        for mid in M:
            dir_1 = []
            for short in S:
                dir_1.append(fr"C:\Users\PESL_RTDS\PycharmProjects\EV_Paper_Combine\DefusionFormer\results_hyper_Sequence\Multi_Input_DeFusionformer_('Multi_Input',)_ftS_sl{Long}_sl{mid}_sl{short}_ll24_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebfixed_dtTrue_test_0")
            dir_Coll.append(np.array(dir_1))
        dir_last.append(np.array(dir_Coll))
    data_last = []
    for j in range(8):
        data_collect = []
        for k in range(len(dir_Coll)):
            data_result = []
            for i in range(len(dir_Coll)):
                data_result.append(np.load(f'{dir_last[j][k][i]}/metrics.npy')[0])
            data_collect.append(np.array(data_result))
        data_last.append(np.array(data_collect))

    return np.array(data_last)

if __name__ == '__main__':
    L = [420, 384, 348, 312, 276, 240, 204, 168]
    M = [48, 60, 72, 84, 96, 108, 120, 132]
    S = [3, 6, 9, 12, 15, 18, 21, 24]
    data = heatmap_plot(L,M, S)
    import os
    os.getcwd()
    np.max(data)
    Hyper_1 = pd.DataFrame(data[0], index=M, columns=S)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', size=15)
    plt.figure(num=1, figsize=(18,8))
    k = 0
    for i in range(7,-1,-1):
        Hyper_1 = pd.DataFrame(data[i], index=M, columns=S)
        plt.subplot(2,4,k+1)
        plt.title(f'Long term = {L[i]}')
        sns.heatmap(Hyper_1,cmap='Blues', vmin=0.81, vmax=0.895)
        k+=1
    plt.tight_layout()
    plt.show()
    #