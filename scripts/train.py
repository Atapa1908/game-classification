"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os

import torch

from torchvision.transforms import v2

import data_setup, engine, model_builder, utils


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/game_pictures/train"
test_dir = "data/game_pictures/test"

# Setup target device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Create transforms
train_transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Resize(size=(64, 64)),
    v2.TrivialAugmentWide(num_magnitude_bins=31), # trivial augment is an augmentation process whereby diversity is introduced to the data, with the hopes of making the model more generalizable
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.RandomHorizontalFlip(p=0.5)
])

test_transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Resize(size=(64, 64)),
    v2.ToDtype(torch.float32, scale=True)  # Normalize expects float input
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transforms,
    test_transform=test_transforms,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_layer=3,
    hidden_layer=HIDDEN_UNITS,
    output_layer=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
