import dataclasses

import json_utils


@dataclasses.dataclass
class TrainConfig:
    learning_rate: float
    weight_decay: float
    dropout: float
    hidden_size: int
    num_layers: int
    num_heads: int
    batch_size: int

    def to_file(self, path):
        json_utils.to_file(dataclasses.asdict(self), path)

    @classmethod
    def from_file(cls, path):
        data = json_utils.from_file(path)
        return cls(**data)


TRAIN_CONFIG_PATH = "train_config/default.json"
TRAIN_CONFIG = TrainConfig.from_file(TRAIN_CONFIG_PATH)
