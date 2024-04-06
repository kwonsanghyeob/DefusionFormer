import numpy as np
import pandas as pd



def compare_Decomposion(num,model,pl):
    dir_last = []
    for NUM in num:
        dir_coll = []
        for PL in pl:
            dir_1 = []
            for MODEL in model:
                dir_1.append(fr"C:\Users\PESL_RTDS\PycharmProjects\EV_Paper_Combine\DefusionFormer\results_Decompostion\test_{NUM}_{MODEL}_('Multi_Input',)_ftS_sl312_sl72_sl9_ll24_pl{PL}_dm512_nh8_el2_dl1_df2048_fc1_ebfixed_dtTrue_test_0")
            dir_coll.append(np.array(dir_1))
        dir_last.append(np.array(dir_coll))
    data_last = []
    for j in range(len(dir_last)):
        data_collect = []
        for k in range(len(dir_last[0])):
            data_result = []
            for i in range(len(dir_last[0][0])):
                data_result.append(np.load(f'{dir_last[j][k][i]}/metrics.npy')[[0,1,3]])
            data_collect.append(np.array(data_result))
        data_last.append(np.array(data_collect))

    return np.array(data_last)

if __name__ == '__main__':
    num = np.arange(0, 9)
    pl = [1, 6, 12, 24, 48]
    model = ['Fusionformer', 'DeFusionformer']

    a = compare_Decomposion(num, model, pl)

    new_data = pd.DataFrame([])
    for i in range(8):
        for j in range(len(pl)):
            data = dict()
            data['Mae'] = a[i][j][:, 0]
            data['Mse'] = a[i][j][:, 1]
            data['nRMSE'] = a[i][j][:, 2]
            data['Model'] = model
            data['station'] = f'data_{i}'
            data['Prediction_length'] = pl[j]

            data_1 = pd.DataFrame(data)
            new_data = pd.concat([new_data, data_1])

    results = pd.DataFrame(np.array(new_data),columns=new_data.columns)

    results_new = results[results['station'] == 'data_1']
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,8))
    sns.catplot(data = results_new, x = 'Model', y='Mae', col= 'Prediction_length', kind="bar")
    plt.show()

    print(new_data)