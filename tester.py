from scripts import data_setup
from scripts import model_builder

import torch
from torchvision.transforms import v2

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


train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir="data/game_pictures/train",
                                                                       test_dir="data/game_pictures/test",
                                                                       train_transform=train_transforms,
                                                                       test_transform=test_transforms,
                                                                       batch_size=32,
                                                                       num_workers=0)

model_test = model_builder.TinyVGG(input_layer=3,
                                   hidden_layer=10,
                                   output_layer=len(class_names))

# dummy forward pass
img_batch, label_batch = next(iter(train_loader))

img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_test.eval()
with torch.inference_mode():
    pred = model_test(img_single)
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")
