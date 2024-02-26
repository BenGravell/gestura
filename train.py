"""Train a model with checkpoints at every epoch."""

import argparse
import logging

import torch
import torch.nn as nn

import dataloader
import modeling
from train_config import TRAIN_CONFIG


def main(num_epochs: int, resume: bool, update_optimizer: bool, reset_optimizer: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_train, dataloader_test = dataloader.load_dataset()

    criterion = nn.CrossEntropyLoss()

    if resume:
        model_path = modeling.get_most_recent_model_path()
        optimizer_path = modeling.get_most_recent_optimizer_path()
        checkpoint = modeling.Checkpoint.from_files(model_path, optimizer_path)
    else:
        checkpoint = modeling.Checkpoint.from_default()

    model = checkpoint.model
    optimizer = checkpoint.optimizer

    if reset_optimizer:
        optimizer = modeling.create_default_optimizer(model)

    if update_optimizer:
        for param_group in optimizer.param_groups:
            param_group["lr"] = TRAIN_CONFIG.learning_rate
            param_group["weight_decay"] = TRAIN_CONFIG.weight_decay

    def evaluate_accuracy(model, dataloader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for feature, target in dataloader:
                feature, target = feature.to(device), target.to(device)
                outputs = model(feature)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
        accuracy = correct / len(dataloader.dataset)
        return accuracy

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(dataloader_train):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Loss and backward pass
            loss = criterion(outputs, targets)
            loss.backward()

            # Update the weights
            optimizer.step()

            # Show the batch training loss for every batch
            logging.info(f"Epoch: {epoch:3d} | Batch: {batch_idx:3d} | Loss: {loss.item():.3f}")

        # Show the train and test accuracy at the end of every epoch
        train_accuracy = evaluate_accuracy(model, dataloader_train)
        test_accuracy = evaluate_accuracy(model, dataloader_test)
        logging.info(f"Epoch: {epoch:3d} | -- Train Accuracy: {train_accuracy:.3f}")
        logging.info(f"Epoch: {epoch:3d} | --  Test Accuracy: {test_accuracy:.3f}")

        # Save a checkpoint
        checkpoint.to_files()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest model.",
    )
    parser.add_argument(
        "--update_optimizer",
        action="store_true",
        help="Configure the optimizer using the training config currently in the repo.",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Reset the optimizer.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs.",
    )
    args = parser.parse_args()
    main(args.num_epochs, args.resume, args.update_optimizer, args.reset_optimizer)
