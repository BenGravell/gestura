"""Utilities."""

import hashlib
from datetime import datetime
import pathlib
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from aeon.datasets import load_classification


def make_dir_if_it_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")


ROOT_TMP_DIR = tempfile.gettempdir()
make_dir_if_it_does_not_exist(tempfile.gettempdir())


NUM_CLASSES = 8

# Assume input sequence x of shape (batch_size, sequence_length, input_size)
batch_size = 32
sequence_length = 315
input_size = 3
hidden_size = 256
output_size = NUM_CLASSES
num_layers = 2
heads = 4

learning_rate = 0.0001


class SklearnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# UWaveGestureLibrary
# http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary
def load_data(dataset_name="UWaveGestureLibrary"):
    extract_path = os.path.join(ROOT_TMP_DIR, "aeon", "datasets")
    # It is critical for extract_path to end with "/", otherwise it will fail.
    # A PR was opened to correct this: https://github.com/aeon-toolkit/aeon/pull/679
    # Once the package is updated with the fix this logic can be simplified.
    if not extract_path.endswith("/"):
        extract_path += "/"
    X, y, meta_data = load_classification(dataset_name, extract_path=extract_path)
    return X, y, meta_data


def load_dataset(dataset_name="UWaveGestureLibrary"):
    X, y, meta_data = load_data(dataset_name)

    # swap axes so data is organized with dims (num_examples, sequence_length, input_size)
    X = np.swapaxes(X, 1, 2)

    # convert to ints
    y = y.astype(float).astype(int)
    # start at zero for easy indexing
    y = y - min(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.uint8)

    # Split the data into a training set and a test set
    with open("train_test_idxs.json", "r") as file_in:
        train_test_idxs = json.load(file_in)
    train_idxs = train_test_idxs["idxs_train"]
    test_idxs = train_test_idxs["idxs_test"]
    X_train, y_train = X_tensor[train_idxs], y_tensor[train_idxs]
    X_test, y_test = X_tensor[test_idxs], y_tensor[test_idxs]

    dataset_train = SklearnDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = SklearnDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataset_train, dataloader_train, dataset_test, dataloader_test


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be evenly divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        # The dot-product is carried out with einstein notation summation.
        # The legend for letter names is:
        # n: number of examples, corresponding to N
        # h: number of heads, corresponding to self.heads
        # d: size of input to each head, corresponding to self.head_dim
        # q: length of query, corresponding to query_len
        # k: length of keys, corresponding to key_len
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out, attention


class LSTMWithAttention(nn.Module):
    def __init__(self):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SelfAttention(embed_size=hidden_size, heads=heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward_with_attn(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, mask)

        # Extract the output at the final timestep
        final_out = attn_out[:, -1, :]

        out = self.fc(final_out)

        return out, attn_weights

    def forward(self, x, mask=None):
        return self.forward_with_attn(x, mask)[0]


def get_timestamp_millis():
    return int(datetime.timestamp(datetime.now()) * 1000)


MODEL_DIR_PATH = pathlib.Path("model")
OPTIM_DIR_PATH = pathlib.Path("optimizer")


def get_most_recent_model_path():
    filenames = [x.name for x in MODEL_DIR_PATH.glob("*.pt")]
    timestamps = [int(filename.split("_")[1]) for filename in filenames]
    latest_timestamp_idx = np.argmax(timestamps)
    latest_timestamp_filename = filenames[latest_timestamp_idx]
    return MODEL_DIR_PATH / latest_timestamp_filename


def get_most_recent_optimizer_path():
    filenames = [x.name for x in OPTIM_DIR_PATH.glob("*.pt")]
    timestamps = [int(filename.split("_")[1]) for filename in filenames]
    latest_timestamp_idx = np.argmax(timestamps)
    latest_timestamp_filename = filenames[latest_timestamp_idx]
    return OPTIM_DIR_PATH / latest_timestamp_filename


def save_checkpoint(model, optimizer):
    model_hash = hashlib.md5(str(model.state_dict()).encode()).hexdigest()
    optimizer_hash = hashlib.md5(str(optimizer.state_dict()).encode()).hexdigest()

    timestamp = get_timestamp_millis()

    model_file_name = f"model_{timestamp}_{model_hash}.pt"
    optimizer_file_name = f"optimizer_{timestamp}_{optimizer_hash}.pt"

    model_file_path = f"model/{model_file_name}"
    optimizer_file_path = f"optimizer/{optimizer_file_name}"

    torch.save(model.state_dict(), model_file_path)
    print(f"Wrote model to {model_file_path}")

    torch.save(optimizer.state_dict(), optimizer_file_path)
    print(f"Wrote optmizer to {optimizer_file_path}")


def load_checkpoint(model_path=None, optimizer_path=None, do_load_optimizer=True):
    model = LSTMWithAttention()
    if model_path is None:
        print("No model path provided, using default model")
    else:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    if do_load_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_path is None:
            print("No optimizer path provided, using default optimizer")
        else:
            optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"Loaded optimizer from {optimizer_path}")
    else:
        optimizer = None

    return {"model": model, "optimizer": optimizer}
