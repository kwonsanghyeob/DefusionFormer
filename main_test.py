import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from former_based import TFModel
from Input_package import windowDataset_test

def evaluate(length):
    input = torch.tensor(data_train[-24 * 7 * 2:]).reshape(1, -1, 1).to(device).float().to(device)
    output = torch.tensor(data_train[-1].reshape(1, -1, 1)).float().to(device)
    model.eval()
    for i in range(length):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        predictions = model(input, output, src_mask, tgt_mask).transpose(0, 1)
        predictions = predictions[:, -1:, :]
        output = torch.cat([output, predictions.to(device)], axis=1)
    return torch.squeeze(output, axis=0).detach().cpu().numpy()[1:]


if __name__ == '__main__':

    device = torch.device("cuda")

    ##데이터 불러오기 및 입력데이터 정의
    rawdata = pd.read_csv("서인천IC-부평IC 평균속도.csv", encoding='CP949')
    plt.figure(figsize=(20, 5))
    plt.plot(range(len(rawdata)), rawdata["평균속도"])
    rawdata.head()

    min_max_scaler = MinMaxScaler()
    rawdata["평균속도"] = min_max_scaler.fit_transform(rawdata["평균속도"].to_numpy().reshape(-1, 1))

    train = rawdata[:-24 * 7]
    data_train = train["평균속도"].to_numpy()

    test = rawdata[-24 * 7:]
    data_test = test["평균속도"].to_numpy()

    iw = 24 * 14
    ow = 24 * 7

    train_dataset = windowDataset_test(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=64)

    val_dataset = windowDataset_test(data_train, input_window=iw, output_window=ow, stride=1)
    val_loader = DataLoader(train_dataset, batch_size=64)

    ##데이터 불러오기 및 입력데이터 정의
    lr = 1e-3
    model = TFModel(256, 8, 256, 2, 1,1, 0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 2

    model.train()
    for i in tqdm(range(epoch)):

        batchloss = 0.0

        for (inputs, dec_inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            tgt_mask = model.generate_square_subsequent_mask(dec_inputs.shape[1]).to(device)

            train_result = model(inputs.float().to(device), dec_inputs.float().to(device), src_mask, tgt_mask)
            loss = criterion(train_result.permute(1, 0, 2), outputs.float().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batchloss += loss

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, dec_inputs, outputs in val_loader:
                src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                tgt_mask = model.generate_square_subsequent_mask(dec_inputs.shape[1]).to(device)
                val_result = model(inputs.float().to(device), dec_inputs.float().to(device), src_mask, tgt_mask)
                val_loss += criterion(val_result.permute(1, 0, 2), outputs.to(device))
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {i + 1}/{epoch}, training Loss: {batchloss}, Validation Loss: {avg_val_loss}")

    result = evaluate(24 * 7)
    result = min_max_scaler.inverse_transform(result)
    real = rawdata["평균속도"].to_numpy()
    real = min_max_scaler.inverse_transform(real.reshape(-1, 1))

    plt.figure(figsize=(20, 5))
    plt.plot(range(400, 744), real[400:], label="real")
    plt.plot(range(744 - 24 * 7, 744), result, label="predict")
    plt.legend()
    plt.show()
