import torch
from torch import nn
from torch.utils.data import DataLoader
from former_based import TFModel
from Input_package import windowDataset, data_preprocessing
from paper_package import *



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    rawdata = pd.read_csv("./Data/SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv", encoding='CP949')
    individual_station = rawdata[['Unnamed: 0', '종합경기장']]
    tr_l = 7747; va_l = 2582; te_l = int(len(rawdata)-(tr_l+va_l));
    P_data, scaler = data_preprocessing(individual_station, tr_l, va_l, te_l)

    iw = 18; ow = 1;
    train_data =  P_data[:tr_l]
    val_data = P_data[tr_l:(tr_l+va_l)]
    test_data = P_data[-te_l:]

    train_dataset = windowDataset(train_data, sequence_length=iw, predict_length=ow, stride=1)
    val_dataset = windowDataset(val_data, sequence_length=iw, predict_length=ow, stride=1)
    test_dataset = windowDataset(test_data, sequence_length=iw, predict_length=ow, stride=1)

    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda")
    lr = 1e-3
    model = TFModel(256, 8, 256, 2, 34,34,0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_loader.dataset.x

    epoch = 1000
    model.train()
    for i in range(epoch):

        batchloss = 0.0

        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            #내가 짠코드
            dec_inp = torch.zeros([inputs.shape[0], ow, inputs.shape[-1]]).float()
            dec_inp = torch.cat([inputs[:, :ow, :]], dim=1).float().to(device)

            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            tgt_mask = model.generate_square_subsequent_mask(dec_inp.shape[1]).to(device)

            train_result = model(inputs.float().to(device), dec_inp, src_mask, tgt_mask)
            loss = criterion(train_result.permute(1, 0, 2), outputs.float().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batchloss += loss

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for (inputs, outputs) in val_loader:
                src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                # dec_inp = torch.zeros([inputs.shape[0], ow, inputs.shape[-1]]).float()
                dec_inp = torch.cat([inputs[:, :ow, :]], dim=1).float().to(device)
                tgt_mask = model.generate_square_subsequent_mask(dec_inp.shape[1]).to(device)
                val_result = model(inputs.float().to(device), dec_inp, src_mask, tgt_mask)
                val_loss += criterion(val_result.permute(1, 0, 2), outputs.to(device))
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {i + 1}/{epoch}, training Loss: {batchloss}, Validation Loss: {avg_val_loss}")

    model.eval()

    preds = []
    trues = []
    for (inputs, outputs) in test_loader:
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)

        # dec_inp = torch.zeros([inputs.shape[0], ow, inputs.shape[-1]]).float()
        dec_inp = torch.cat([inputs[:, :ow, :]], dim=1).float().to(device)

        tgt_mask = model.generate_square_subsequent_mask(dec_inp.shape[1]).to(device)
        test_result = model(inputs.float().to(device), dec_inp, src_mask, tgt_mask)

        preds.append(test_result.permute(1, 0, 2))
        trues.append(outputs)

    import numpy as np
    pred = np.array(preds[0].detach().cpu().numpy(), dtype='float').reshape(-1)
    for i in range(1,len(preds)):
        pred = np.concatenate((pred, np.array(preds[i].detach().cpu().numpy(), dtype='float').reshape(-1)))

    true = np.array(trues[0].detach().cpu().numpy(), dtype='float').reshape(-1)
    for i in range(1, len(trues)):
        true = np.concatenate((true, np.array(trues[i].detach().cpu().numpy(), dtype='float').reshape(-1)))


    result = dict()
    result['real'] = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
    result['predict'] = scaler.inverse_transform(true.reshape(-1, 1)).reshape(-1)

    result_pd= pd.DataFrame(result)
    #모델 저장하기
    #데이터
    result_pd.to_csv('transformer_result.csv')
    torch.save(model.state_dict(), 'saved_model/transformer')