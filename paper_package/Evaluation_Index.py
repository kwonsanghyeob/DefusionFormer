import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

##
# def MSE(Actual,Pred):
#     return ((Actual-Pred)**2).mean()
def MSE(Actual,Pred):
    return mean_squared_error(Actual, Pred)

def MAE(Actual,Pred):
    return mean_absolute_error(Actual, Pred)

def RSME(Actual,Pred):
    return mean_squared_error(Actual, Pred)**0.5

def nRSME(Actual,Pred):
    return mean_squared_error(Actual, Pred)**0.5/(max(Actual)-min(Actual))

# def MAE(Actual,Pred):
#     return np.abs(Actual, Pred).mean()

def smape(Actual, Pred):
    return 100 / len(Actual) * np.sum(np.abs(Pred - Actual) / (np.abs(Actual) + np.abs(Pred)))

def masked_mape_np(y_true, y_pred, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

