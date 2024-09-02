import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2 # video uses ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from PIL import Image
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm


# Set device type
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# print(torch.__version__)
# print(torchvision.__version__)
# print(device)


EPOCHS = 50

# img = Image.open("game_pics/board_game/Screenshot 2024-08-23 at 2.06.35 PM.png")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# img.show()

# Setup training data
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

dataset = ImageFolder(root='game_pics', transform=transforms)

print(len(dataset))
class_names = dataset.classes
print(class_names)

class_dict = dataset.class_to_idx
print(class_dict)

print(dataset.samples)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

print(train_size)
print(test_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset.dataset.transform = train_transforms
test_dataset.dataset.transform = test_transforms

# # visualize a random image
# test_image, test_label = train_dataset[0]

# print(test_image.shape)
# print(test_image.dtype)

# img_reshaped = test_image.permute(1, 2, 0) # image shape is changed from (C, H, W) to (H , W, C) because matplotlib needs that format
# print(f"Image shape: {img_reshaped.shape}")
# plt.figure(figsize=(10, 7))
# plt.imshow(img_reshaped) # image shape is [1, 28, 28] (colour channels, height, width)
# plt.axis("off")
# plt.title(class_names[test_label], fontsize=20)
# plt.show()


# turn data into dataloader (an iterable)
print(os.cpu_count()) # nrr of cpu cores used to load data

train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True
)

test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False
)

img_load, label_load = next(iter(train_loader))

print(img_load.shape)
print(label_load.shape)

class TinyVGG(nn.Module):
    def __init__(self, input_layer: int, hidden_layer: int, output_layer: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_layer,
                      out_channels=hidden_layer,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer,
                      out_channels=hidden_layer,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,
                      out_channels=hidden_layer,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer,
                      out_channels=hidden_layer,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_layer*13*13, # after flattening, exact number of inputs needs to be calculated (in this case it's 7*7: `x.shape` after conv_block_2 -> torch.Size([10, 7, 7]))
                  out_features=output_layer)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    
# Instantiate our model from line 101
torch.manual_seed(42)

model_0 = TinyVGG(input_layer=3, # of color channels in image
                  hidden_layer=100,
                  output_layer=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

print(model_0(img_load.to(device))) # need to adjust the last linear layer (after flattening) at this point via error message

print(summary(model_0, input_size=[32, 3, 64, 64]))

# define a `train_step()` and a `test_step()` function

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.dataloader.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    
    loss_train, acc_train = 0, 0
    model.to(device)

    model.train()

    # Add a loop to loop through training batches
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_pred = model(X_train)

        # Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y_train)
        loss_train += loss.item() # accumulate loss_train

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc_train += (y_pred_class==y_train).sum().item()/len(y_pred)

    loss_train /= len(dataloader)
    acc_train /= len(dataloader)

    return loss_train, acc_train

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.dataloader.DataLoader,
              loss_fn: torch.nn.Module):

    model.eval()

    loss_test, acc_test = 0, 0

    # Turn on inference mode for evaluating
    with torch.inference_mode():

    # Loop through test dataloader batches
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Forward pass
            y_pred_test_logits = model(X_test)

            # Calculate loss and accuracy (per batch)
            loss = loss_fn(y_pred_test_logits, y_test)
            loss_test += loss.item() # accumulate loss_train

            y_pred_class_test = torch.argmax(torch.softmax(y_pred_test_logits, dim=1), dim=1)
            acc_test += (y_pred_class_test==y_test).sum().item()/len(y_pred_test_logits)

    loss_test /= len(dataloader)
    acc_test /= len(dataloader)

    return loss_test, acc_test


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# start timer
start_time = timer()

# train our model_0
model_0_results = train(model_0,
               train_dataloader=train_loader,
               test_dataloader=test_loader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=EPOCHS)

# end timer
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")


image_to_test = torchvision.io.read_image("Screenshot 2024-08-27 at 3.00.51 PM.png")
if image_to_test.shape[0] == 4:
    image_to_test = image_to_test[:3]

# plt.imshow(image_to_test.permute(1, 2, 0))
# plt.show()

image_transform = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Resize(size=(64, 64)),
    v2.ToDtype(torch.float32, scale=True)
])

image_to_test_transformed = image_transform(image_to_test)

print(image_to_test_transformed.dtype)

# plt.imshow(image_to_test_transformed.permute(1, 2, 0))
# plt.show()

# make a prediction on the loaded test image
model_0.eval()
with torch.inference_mode():
    custom_image_pred = model_0(image_to_test_transformed.unsqueeze(0).to(device)) # need to add a batch dimenstion with `unsqueeze()`

print(custom_image_pred) # these are logits; convert them to probabilities and then to label next

custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1) # converts to prediction probabilities

custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

print(class_names[custom_image_pred_label])


# save our model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "game_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

MODEL_SAVE_PATH

# Saving the model
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
