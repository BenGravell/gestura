"""Model definitions."""

import logging
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

import constants
import paths
import utils
from train_config import TRAIN_CONFIG


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        if not self.head_dim * num_heads == embed_size:
            msg = "Embedding size must be evenly divisible by num_heads"
            raise RuntimeError(msg)

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        # The dot-product is carried out with einstein notation summation.
        # The legend for letter names is:
        # n: number of examples, corresponding to N
        # h: number of heads, corresponding to self.num_heads
        # d: size of input to each head, corresponding to self.head_dim
        # q: length of query, corresponding to query_len
        # k: length of keys, corresponding to key_len
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out, attention


class LSTMWithAttention(nn.Module):
    """LSTM with Multi-head Self-Attention."""

    def __init__(self):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(constants.INPUT_SIZE, TRAIN_CONFIG.hidden_size, TRAIN_CONFIG.num_layers, batch_first=True, dropout=TRAIN_CONFIG.dropout)
        self.attention = SelfAttention(embed_size=TRAIN_CONFIG.hidden_size, num_heads=TRAIN_CONFIG.num_heads)
        self.fc = nn.Linear(TRAIN_CONFIG.hidden_size, constants.OUTPUT_SIZE)

    def forward_with_attn(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, mask)

        # Extract the output at the final timestep
        final_out = attn_out[:, -1, :]

        out = self.fc(final_out)

        return out, attn_weights

    def forward(self, x, mask=None):
        return self.forward_with_attn(x, mask)[0]


def create_default_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG.learning_rate, weight_decay=TRAIN_CONFIG.weight_decay)


@dataclasses.dataclass
class Checkpoint:
    model: LSTMWithAttention
    optimizer: Optimizer

    def to_files(self):
        model_hash = utils.get_hash(self.model.state_dict())
        optimizer_hash = utils.get_hash(self.optimizer.state_dict())

        timestamp = utils.get_timestamp_millis()

        model_file_name = f"model_{timestamp}_{model_hash}.pt"
        optimizer_file_name = f"optimizer_{timestamp}_{optimizer_hash}.pt"

        model_file_path = paths.MODEL_DIR_PATH / model_file_name
        optimizer_file_path = paths.OPTIM_DIR_PATH / optimizer_file_name

        torch.save(self.model.state_dict(), model_file_path)
        logging.info(f"Wrote model to {model_file_path}")

        torch.save(self.optimizer.state_dict(), optimizer_file_path)
        logging.info(f"Wrote optmizer to {optimizer_file_path}")

    @classmethod
    def from_default(cls):
        model = LSTMWithAttention()
        optimizer = create_default_optimizer(model)
        return cls(model, optimizer)

    @classmethod
    def from_files(cls, model_path=None, optimizer_path=None):
        checkpoint = cls.from_default()
        if model_path is None:
            logging.info("No model path provided, using default model")
        else:
            checkpoint.model.load_state_dict(torch.load(model_path))
            logging.info(f"Loaded model from {model_path}")

        if optimizer_path is None:
            logging.info("No optimizer path provided, using default optimizer")
        else:
            checkpoint.optimizer.load_state_dict(torch.load(optimizer_path))
            logging.info(f"Loaded optimizer from {optimizer_path}")

        return checkpoint


def get_most_recent_model_path():
    return utils.get_most_recent_path(paths.MODEL_DIR_PATH, glob_pattern="*.pt")


def get_most_recent_optimizer_path():
    return utils.get_most_recent_path(paths.OPTIM_DIR_PATH, glob_pattern="*.pt")
