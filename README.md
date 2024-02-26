# Gestura

Gesture prediction from time-series data.

## Usage

### [required]

1. Run `train.py` to train the model.
2. Run `streamlit run Gestura.py` to visualize the predictions.

### [optional]

1. Run `train_test_split.py` to regenerate the train/test split indices.
2. Run `clean_up_artifacts.py` to remove old model and optimizer checkpoints.

## Development

### Pre-commit

Run

```bash
pre-commit run --all-files
```

to run all pre-commit hooks, including style formatting and unit tests.

### Package management

Update [`requirements.in`](requirements.in) with new direct dependencies.

Then run

```bash
pip-compile requirements.in
```

to update the [`requirements.txt`](requirements.txt) file with all indirect and transitive dependencies.

Then run

```bash
pip install -r requirements.txt
```

to update your virtual environment with the packages.
