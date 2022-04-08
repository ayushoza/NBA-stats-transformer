import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import pandas as pd
import ast
import numpy as np
import math
import matplotlib.pyplot as plt
import joblib

torch.manual_seed(1)


## code from "Language Modeling with nn.Transformer and TorchText"
## https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True,
                                                 norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5795):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len*d_model).reshape(max_len, d_model, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


train = pd.read_csv("data/data_training_standardized.csv", header=None)
valid = pd.read_csv("data/data_validation_standardized.csv", header=None)
test = pd.read_csv("data/data_testing_standardized.csv", header=None)


def process_data(file):
    seq = []
    b = []
    for j in range(len(file)):
        for i in range(len(file.loc[j])):
            s = " ".join(file.loc[j][i].split()).replace(
                " ", ",").replace("[,", "[").replace(",]", "]")
            full_career = ast.literal_eval(s)
            seq.append(full_career)
    tt = torch.tensor(seq).reshape([len(file), len(file.loc[0]), 13])
    tt_input = tt[:, :12, :12]
    tt_output = tt[:, 12:, :12]
    return tt_input, tt_output


# transformer dimensions
ntoken = 12
d_model = 12
nhead = 6
d_hid = 12
nlayers = 3
dropout = 0.1

# processed data inputs and labels
tt_input, tt_output = process_data(train)
val_input, val_output = process_data(valid)
test_input, test_output = process_data(test)
indices = torch.randperm(tt_input.size(0))
shuffled_inp, shuffled_out = tt_input[indices], tt_output[indices]


def get_batch(data_input: Tensor, data_output: Tensor, batch_size: int):
    """
    Create batches for the data
    """
    inp_batches = []
    out_batches = []
    seq_len = data_input.size(0) // batch_size
    rem = data_input.size(0) % batch_size
    for i in range(seq_len):
        batch_inp = data_input[i*batch_size:(i+1)*batch_size]
        batch_out = data_output[i*batch_size:(i+1)*batch_size]
        inp_batches.append(batch_inp)
        out_batches.append(batch_out)
    inp_batches.append(data_input[(i+1)*batch_size: ((i+1)*batch_size)+rem])
    out_batches.append(data_output[(i+1)*batch_size: ((i+1)*batch_size)+rem])

    return inp_batches, out_batches


# initialize model
transf_model = TransformerModel(
    ntoken, d_model, nhead, d_hid, nlayers, dropout)


# train the model
def train(lr, batch_size, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transf_model.parameters(), lr=lr)

    loss_vals, loss_vals_val = [], []
    first_loss_vals, second_loss_vals = [], []
    for i in range(epochs):
        indices = torch.randperm(tt_input.size(0))
        shuffled_inp, shuffled_out = tt_input[indices], tt_output[indices]
        d_input, d_output = get_batch(shuffled_inp, shuffled_out, batch_size)
        remainder = tt_input.size(0) % batch_size
        t_loss = 0
        first_season_loss, second_season_loss = 0, 0
        for j in range(len(d_input)):
            optimizer.zero_grad()
            d_input[j].requires_grad_(True)
            if j == len(d_input) - 1:
                output = transf_model(
                    d_input[j], torch.zeros([nhead*remainder, 12, 12]))
            else:
                output = transf_model(
                    d_input[j], torch.zeros([nhead*batch_size, 12, 12]))
            loss = criterion(output, d_output[j])
            first_season_loss += criterion(output[:,:6,:], d_output[j][:,:6,:])
            second_season_loss += criterion(output[:,6:,:], d_output[j][:,6:,:])
            t_loss += loss
            loss.backward()
            optimizer.step()
        t_loss /= len(d_input)
        first_season_loss /= len(d_input)
        second_season_loss /= len(d_input)
        v_loss = criterion(val_output, transf_model(
            val_input, torch.zeros([nhead*val_input.size(0), 12, 12])))
        loss_vals.append(float(t_loss.detach()))
        loss_vals_val.append(float(v_loss.detach()))
        first_loss_vals.append(float(first_season_loss.detach()))
        second_loss_vals.append(float(second_season_loss.detach()))

    plt.figure()
    plt.plot(loss_vals, "-b", label="train")
    plt.plot(loss_vals_val, "-r", label="validation")
    plt.legend(loc="upper right")
    plt.title("Transformer Model")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.savefig("losses.png")
    # plt.show()
    
    plt.figure()
    plt.plot(first_loss_vals, "-c", label="first predicted season")
    plt.plot(second_loss_vals, "-m", label="second predicted season")
    plt.legend(loc="upper right")
    plt.title("First vs. Second Seasons Predicted")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.savefig("first_vs_second.png")


train(lr=0.0005, batch_size=64, epochs=500)
crit = nn.MSELoss()
print(crit(test_output, transf_model(
    test_input, torch.zeros([nhead*test_input.size(0), 12, 12]))))


joblib.dump(transf_model, "./transformer_model.joblib", compress=True)

