import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding


class DeFusionformer_M(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(DeFusionformer_M, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        #Decomposition
        # Decomp
        kernel_size_L = configs.moving_avg_L
        kernel_size_M = configs.moving_avg_M
        kernel_size_S = configs.moving_avg_S

        self.decomp_L = series_decomp(kernel_size_L)
        self.decomp_M = series_decomp(kernel_size_M)
        self.decomp_S = series_decomp(kernel_size_S)

        self.Weight_L = nn.Linear(1, configs.d_model)
        self.Weight_M = nn.Linear(1, configs.d_model)
        self.Weight_S = nn.Linear(1, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_L_enc, x_L_mark_enc, x_M_enc, x_M_mark_enc, x_S_enc, x_S_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # decomp init_mid
        seasonal_M_init, trend_M_init = self.decomp_M(x_M_enc)

        weight_M = self.Weight_M(trend_M_init)

        enc_M_out = self.enc_embedding(x_M_enc, x_M_mark_enc)
        enc_M_out, _ = self.encoder(enc_M_out, attn_mask=enc_self_mask)

        enc_mid = weight_M + enc_M_out

        enc = torch.concat([enc_mid], dim=1)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], _
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

if __name__ == '__main__':
    # model define
    import argparse

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg_L', type=int, default=25, help='window size of moving average')
    parser.add_argument('--moving_avg_M', type=int, default=15, help='window size of moving average')
    parser.add_argument('--moving_avg_S', type=int, default=5, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args = parser.parse_args()
    '''
    batch_x_L = [32, 300, 1]
    batch_x_L_mark = [32, 300, 4]
    batch_x_M = [32, 200, 1]
    batch_x_M_mark = [32, 200, 4]
    batch_x_S = [32, 100, 1]
    batch_x_S_mark = [32, 100, 4]
    batch_y = [32, 12, 1]
    batch_y_mark = [32, 12, 4]
    '''
    device = torch.device("cuda")
    batch_x_L = torch.rand(32, 300, 1).float().to(device)
    batch_x_L_mark = torch.rand(32, 300, 4).float().to(device)
    batch_x_M = torch.rand(32, 200, 1).float().to(device)
    batch_x_M_mark = torch.rand(32, 200, 4).float().to(device)
    batch_x_S = torch.rand(32, 100, 1).float().to(device)
    batch_x_S_mark = torch.rand(32, 100, 4).float().to(device)
    batch_y = torch.rand(32, 12, 1).float().to(device)
    batch_y_mark = torch.rand(32, 12, 4).float().to(device)

    a = torch.zeros(1,1,3)
    b = torch.ones(1, 1, 1)
    c = a+b
    a = DeFusionformer(args)
    a(batch_x_L, batch_x_L_mark,
      batch_x_M, batch_x_M_mark,
      batch_x_S, batch_x_S_mark,
      batch_y, batch_y_mark)

