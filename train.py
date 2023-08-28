"""Train a model with checkpoints."""

import argparse

import torch
import torch.nn as nn

import utils


def main(num_epochs=10, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, dataloader_train, _, dataloader_test = utils.load_dataset()

    criterion = nn.CrossEntropyLoss()

    if resume:
        model_path = utils.get_most_recent_model_path()
        optimizer_path = utils.get_most_recent_optimizer_path()
        checkpoint = utils.load_checkpoint(model_path, optimizer_path)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
    else:
        model = utils.LSTMWithAttention()
        optimizer = torch.optim.Adam(model.parameters(), lr=utils.learning_rate)

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
            print(f"Epoch: {epoch:3d} | Batch: {batch_idx:3d} | Loss: {loss.item():.3f}")

        # Show the train and test accuracy at the end of every epoch
        train_accuracy = evaluate_accuracy(model, dataloader_train)
        test_accuracy = evaluate_accuracy(model, dataloader_test)
        print(f"Epoch: {epoch:3d} | -- Train Accuracy: {train_accuracy:.3f}")
        print(f"Epoch: {epoch:3d} | --  Test Accuracy: {test_accuracy:.3f}")
        utils.save_checkpoint(model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup the model and optimizer directories to remove obsolete data.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest model.",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.",)
    args = parser.parse_args()
    main(args.num_epochs, args.resume)
