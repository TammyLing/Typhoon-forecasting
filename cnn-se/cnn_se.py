import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
import sys
import random


def random_flip_and_rotate(tensor, seed):
    """
    Randomly flip and rotate a tensor.

    Args:
    - tensor: A tensor of shape (C, H, W).
    - seed: A random seed.

    Returns:
    - A tensor of shape (C, H, W) that is randomly flipped and rotated.
    """
    random.seed(seed)
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[1])
    rotations = random.choice([0, 1, 2, 3])
    tensor = torch.rot90(tensor, k=rotations, dims=[1, 2])
    return tensor

class TyphoonDataset(Dataset):
    """
    A Dataset that provides access to the Ditigal Typhoon dataset and ERA5 dataset.
    """
    def __init__(self, dataset, transform=None, augment=False):
        super().__init__()
        self.dataset = dataset
        
        # get the image data and the target data
        self.image_data = self.dataset['image_data'].values
        self.u10 = self.dataset['u10'].values
        self.v10 = self.dataset['v10'].values
        self.sp = self.dataset['sp'].values
        self.t2m = self.dataset['t2m'].values

        self.transform = transform
        self.augment = augment

        # calculate min and max values for normalization
        self.u10_min = np.min(self.u10[:])
        self.u10_max = np.max(self.u10[:])
        self.v10_min = np.min(self.v10[:])
        self.v10_max = np.max(self.v10[:])
        self.sp_min = np.min(self.sp[:])
        self.sp_max = np.max(self.sp[:])
        self.t2m_min = np.min(self.t2m[:])
        self.t2m_max = np.max(self.t2m[:])

        self.image_min = np.min(self.image_data[:])
        self.image_max = np.max(self.image_data[:])

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, index):
        # get the image and target data
        img = self.image_data[index]

        # flip the image vertically to match the ERA5 data
        img = np.flipud(img)

        # resize the image to 64x64
        img_64 = np.array(Image.fromarray(img).resize((64, 64), Image.BICUBIC))

        # normalize the image data
        img = img.astype(np.float32)
        img = (img - self.image_min) / (self.image_max - self.image_min)  
        img = torch.tensor(img).unsqueeze(0) 

        img_64 = img_64.astype(np.float32)
        img_64 = (img_64 - self.image_min) / (self.image_max - self.image_min) 
        img_64 = torch.tensor(img_64).unsqueeze(0) 
        img_64 = img_64.repeat(4, 1, 1)

        if self.transform:
            img_64 = self.transform(img_64)

        # normalize the target data
        u10 = np.array(Image.fromarray(self.u10[index]).resize((64, 64), Image.BICUBIC))
        u10 = (u10 - self.u10_min) / (self.u10_max - self.u10_min)
        u10 = torch.tensor(u10, dtype=torch.float32).unsqueeze(0)

        v10 = np.array(Image.fromarray(self.v10[index]).resize((64, 64), Image.BICUBIC))
        v10 = (v10 - self.v10_min) / (self.v10_max - self.v10_min)
        v10 = torch.tensor(v10, dtype=torch.float32).unsqueeze(0)

        sp = np.array(Image.fromarray(self.sp[index]).resize((64, 64), Image.BICUBIC))
        sp = (sp - self.sp_min) / (self.sp_max - self.sp_min)
        sp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

        t2m = np.array(Image.fromarray(self.t2m[index]).resize((64, 64), Image.BICUBIC))
        t2m = (t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        t2m = torch.tensor(t2m, dtype=torch.float32).unsqueeze(0)

        original_u10 = self.u10[index].astype(np.float32)
        original_u10 = (original_u10 - self.u10_min) / (self.u10_max - self.u10_min)
        original_u10 = torch.tensor(original_u10, dtype=torch.float32).unsqueeze(0)

        original_v10 = self.v10[index].astype(np.float32)
        original_v10 = (original_v10 - self.v10_min) / (self.v10_max - self.v10_min)
        original_v10 = torch.tensor(original_v10, dtype=torch.float32).unsqueeze(0)

        original_sp = self.sp[index].astype(np.float32)
        original_sp = (original_sp - self.sp_min) / (self.sp_max - self.sp_min)
        original_sp = torch.tensor(original_sp, dtype=torch.float32).unsqueeze(0)

        original_t2m = self.t2m[index].astype(np.float32)
        original_t2m = (original_t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        original_t2m = torch.tensor(original_t2m, dtype=torch.float32).unsqueeze(0)

        # data augmentation
        if self.augment:
            seed = np.random.randint(2147483647)
            img_64 = random_flip_and_rotate(img_64, seed)
            u10 = random_flip_and_rotate(u10, seed)
            v10 = random_flip_and_rotate(v10, seed)
            sp = random_flip_and_rotate(sp, seed)
            t2m = random_flip_and_rotate(t2m, seed)
            original_u10 = random_flip_and_rotate(original_u10, seed)
            original_v10 = random_flip_and_rotate(original_v10, seed)
            original_sp = random_flip_and_rotate(original_sp, seed)
            original_t2m = random_flip_and_rotate(original_t2m, seed)

        return img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual Block that contains two convolutional layers with batch normalization and dropout.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.downsample = nn.Sequential()
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        if self.use_se:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ImprovedCNN(nn.Module):
    """
    Improved CNN model that contains residual blocks with squeeze-and-excitation blocks.
    """
    def __init__(self, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()
        self.layer1 = self._make_layer(4, 64, stride=2, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, stride=2, dropout_rate=dropout_rate)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, stride=1, dropout_rate=0.3, use_se=True):
        layer = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, dropout_rate, use_se),
            ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate, use_se=use_se)
        )
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.sigmoid(x)
        return x

def train_cnn(train_loader, val_loader, model, optimizer, device, num_epochs=200, patience=20):
    """
    Train a CNN model on the training set and evaluate it on the validation set.
    
    Args:
    - train_loader: A DataLoader that provides the training set.
    - val_loader: A DataLoader that provides the validation set.
    - model: A CNN model.
    - optimizer: An optimizer for training the model.
    - device: A device to run the model on.
    - num_epochs: The number of epochs to train the model.
    - patience: The number of epochs to wait for the validation loss to improve before early stopping.
    
    Returns:
    - A list of training losses.
    - A list of validation losses.
    """
    # Set the model to training mode
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0

    criteria = [nn.MSELoss() for _ in range(4)]
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0 
        for i, data in enumerate(train_loader):
            img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
            targets = [u10, v10, sp, t2m]
            inputs = img_64.to(device) 
            
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.split(1, dim=1)  
            targets = [t.to(device) for t in targets]

            # Calculate the loss
            weight_u10 = 10.0
            weight_v10 = 10.0
            weight_sp = 1.0
            weight_t2m = 1.0

            loss_u10 = criteria[0](outputs[0], targets[0])
            loss_v10 = criteria[1](outputs[1], targets[1])
            loss_sp = criteria[2](outputs[2], targets[2])
            loss_t2m = criteria[3](outputs[3], targets[3])

            loss = weight_u10 * loss_u10 + weight_v10 * loss_v10 + weight_sp * loss_sp + weight_t2m * loss_t2m

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  

        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)  
        train_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_epoch_loss:.4f}', flush=True)
        sys.stdout.flush()

        val_loss = evaluate_cnn(val_loader, model, device, None)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}', flush=True)
        sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'cnn_se_best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print('Finished Training')
    sys.stdout.flush()

    return train_losses, val_losses

def evaluate_cnn(loader, model, device, dataset):
    """
    Evaluate a CNN model on a dataset.
    
    Args:
    - loader: A DataLoader that provides the dataset.
    - model: A CNN model.
    - device: A device to run the model on.
    - dataset: A dataset to calculate the loss.
    
    Returns:
    - The average loss on the dataset
    """
    # Set the model to evaluation mode
    model.eval()
    running_loss = 0.0
    criteria = [nn.MSELoss() for _ in range(4)]
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
            targets = [u10, v10, sp, t2m]
            inputs = img_64.to(device)  

            outputs = model(inputs)
            outputs = outputs.split(1, dim=1)  
            targets = [t.to(device) for t in targets]

            # Calculate the loss
            weight_u10 = 10.0
            weight_v10 = 10.0
            weight_sp = 1.0
            weight_t2m = 1.0

            loss_u10 = criteria[0](outputs[0], targets[0])
            loss_v10 = criteria[1](outputs[1], targets[1])
            loss_sp = criteria[2](outputs[2], targets[2])
            loss_t2m = criteria[3](outputs[3], targets[3])

            loss = weight_u10 * loss_u10 + weight_v10 * loss_v10 + weight_sp * loss_sp + weight_t2m * loss_t2m

            running_loss += loss.item()

    return running_loss / len(loader)

def test_cnn_visual(loader, model, device, dataset):
    """
    Visualize the prediction of a CNN model on a test sample.
    
    Args:
    - loader: A DataLoader that provides the test set.
    - model: A CNN model.
    - device: A device to run the model on.
    - dataset: A dataset to calculate the loss.
    
    Returns:
    - None
    """
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
            targets = [u10, v10, sp, t2m]
            inputs = img_64.to(device) 

            outputs = model(inputs)
            outputs = outputs.split(1, dim=1) 
            targets = [t.to(device) for t in targets]

            if i == 0:
                for j, target_name in enumerate(['U10', 'V10', 'Pressure', 'Temperature']):
                    prediction = outputs[j][0].cpu().numpy().squeeze()
                    true_value = targets[j][0].cpu().numpy().squeeze()
                    
                    # denormalize the prediction and true value
                    if target_name == 'U10':
                        prediction = prediction * (dataset.u10_max - dataset.u10_min) + dataset.u10_min
                        true_value = true_value * (dataset.u10_max - dataset.u10_min) + dataset.u10_min
                    elif target_name == 'V10':
                        prediction = prediction * (dataset.v10_max - dataset.v10_min) + dataset.v10_min
                        true_value = true_value * (dataset.v10_max - dataset.v10_min) + dataset.v10_min
                    elif target_name == 'Pressure':
                        prediction = prediction * (dataset.sp_max - dataset.sp_min) + dataset.sp_min
                        true_value = true_value * (dataset.sp_max - dataset.sp_min) + dataset.sp_min
                    else:
                        prediction = prediction * (dataset.t2m_max - dataset.t2m_min) + dataset.t2m_min
                        true_value = true_value * (dataset.t2m_max - dataset.t2m_min) + dataset.t2m_min

                    prediction_resized = np.array(Image.fromarray(prediction).resize((512, 512), Image.BICUBIC))

                    # plot the input image, true value, predicted value, and predicted value resized to 512x512
                    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                    input_plot = axs[0, 0].imshow(img_64[0, 0].cpu().numpy().squeeze(), cmap='jet')
                    axs[0, 0].set_title('Input Image')
                    fig.colorbar(input_plot, ax=axs[0, 0])

                    true_plot = axs[0, 1].imshow(true_value, cmap='jet')
                    axs[0, 1].set_title('True Value')
                    fig.colorbar(true_plot, ax=axs[0, 1])

                    pred_plot = axs[1, 0].imshow(prediction, cmap='jet')
                    axs[1, 0].set_title('Predicted Value')
                    fig.colorbar(pred_plot, ax=axs[1, 0])

                    pred_resized_plot = axs[1, 1].imshow(prediction_resized, cmap='jet')
                    axs[1, 1].set_title('Predicted Resized to 512')
                    fig.colorbar(pred_resized_plot, ax=axs[1, 1])

                    plt.tight_layout()
                    plt.savefig(f'cnn_se_prediction_vs_true_{target_name}_plot_augmentation.png')
                    plt.show()
                    plt.close(fig)
            break

def load_all_zarr_files(directory):
    """
    Load all zarr files in the specified directory and combine them into a single xarray dataset.
    
    Args:
    - directory: The directory that contains the zarr files.
    
    Returns:
    - A combined xarray dataset.
    """
    datasets = []
    # Go through the directory and find all zarr files
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.endswith(".zarr"):
                zarr_path = os.path.join(root, dir)
                print(f"Found zarr directory: {zarr_path}")
                sys.stdout.flush()
                ds = xr.open_zarr(zarr_path)
                datasets.append(ds)
    if not datasets:
        raise ValueError("No zarr files found in the specified directory。")
    combined_dataset = xr.concat(datasets, dim="time")
    return combined_dataset

if __name__ == '__main__':
    # Load the dataset
    base_directory = os.path.join('/vol', 'bitbucket', 'zl1823', 'Typhoon-forecasting','dataset', 'pre_processed_dataset', 'ERA5_without_nan_taiwan')
    combined_dataset = load_all_zarr_files(base_directory)
    
    # split 60% of the data for training, 20% for validation, and 20% for testing
    dataset = TyphoonDataset(combined_dataset)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    augmentation_ratio = 0.5
    augmented_size = int(train_size * augmentation_ratio)
    augmented_dataset = TyphoonDataset(combined_dataset, augment=True)
    augmented_dataset = random_split(augmented_dataset, [augmented_size, len(augmented_dataset) - augmented_size])[0]

    # concatenate the original training dataset and the augmented dataset
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Finetune the CNN-SE model
    print(f"Finetuning CNN-SE Model:")
    batch_sizes = [32, 16, 8]
    learning_rates = [1e-5, 1e-4, 1e-3]
    best_val_loss = float('inf')
    best_params = {}

    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"Training with batch_size={batch_size} and learning_rate={lr}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model = ImprovedCNN(dropout_rate=0.3).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_losses, val_losses = train_cnn(train_loader, val_loader, model, optimizer, device, num_epochs=200)
            
            model.load_state_dict(torch.load('cnn_se_best_model.pth'))
            val_loss = min(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'batch_size': batch_size, 'learning_rate': lr}

    print(f"Best parameters found: {best_params}")

    # Train the CNN-SE model with the best parameters
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # Train the CNN-SE model with the best parameters
    print(f"Testing best CNN-SE Model:")
    model = ImprovedCNN(dropout_rate=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    train_losses, val_losses = train_cnn(train_loader, val_loader, model, optimizer, device, num_epochs=200)

    model.load_state_dict(torch.load('cnn_se_best_model.pth'))
    test_loss = evaluate_cnn(test_loader, model, device, dataset)
    print(f"Test Loss: {test_loss:.4f}")

    test_cnn_visual(test_loader, model, device, dataset)

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
