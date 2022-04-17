import pandas as pd
import numpy as np
import torch, ast
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn as nn
import joblib
from training_final1 import TransformerModel as t


train = pd.read_csv("data/data_training_standardized.csv", header=None)
valid = pd.read_csv("data/data_validation_standardized.csv", header=None)
test = pd.read_csv("data/data_testing_standardized.csv", header=None)


def process_data(file):
    seq = []
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
transf_model = t.TransformerModel(
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