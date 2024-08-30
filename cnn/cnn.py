import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import xarray as xr
import sys
import random

from PIL import Image
from torchvision import transforms as T

def set_seed(seed):
    '''
    Set the random seed for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    set_seed(42 + worker_id)

def random_flip_and_rotate(tensor, seed):
    '''
    Randomly flip and rotate the tensor.

    Args:
    - tensor: the input tensor
    - seed: the random seed

    Returns:
    - the transformed tensor
    '''
    random.seed(seed)
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[1])
    rotations = random.choice([0, 1, 2, 3])
    tensor = torch.rot90(tensor, k=rotations, dims=[1, 2])
    return tensor

def add_noise(tensors, noise_level=0.01):
    '''
    Add Gaussian noise to the input tensors.
    
    Args:
    - tensors: a list of input tensors
    - noise_level: the standard deviation of the Gaussian noise
    
    Returns:
    - a list of noisy tensors
    '''
    noisy_tensors = [tensor + noise_level * torch.randn_like(tensor) for tensor in tensors]
    return noisy_tensors

def apply_smoothing(tensors, kernel_size=3):
    '''
    Apply Gaussian smoothing to the input tensors.
    
    Args:
    - tensors: a list of input tensors
    
    Returns:
    - a list of smoothed tensors
    '''
    smoothed_tensors = [T.GaussianBlur(kernel_size=kernel_size)(tensor) for tensor in tensors]
    return smoothed_tensors

def adjust_contrast(tensors, contrast_factor=0.1):
    '''
    Adjust the contrast of the input tensors.
    
    Args:
    - tensors: a list of input tensors
    - contrast_factor: the factor to adjust the contrast
    
    Returns:
    - a list of contrast-adjusted tensors
    '''
    contrast_tensors = []
    for tensor in tensors:
        channels = tensor.split(1, dim=0)
        adjusted_channels = [T.functional.adjust_contrast(ch, contrast_factor=random.uniform(1-contrast_factor, 1+contrast_factor)) for ch in channels]
        contrast_tensor = torch.cat(adjusted_channels, dim=0)
        contrast_tensors.append(contrast_tensor)
    return contrast_tensors

def conservative_augmentation(tensors, seed):
    '''
    Apply conservative augmentation to the input tensors.
    
    Args:
    - tensors: a list of input tensors
    - seed: the random seed
    
    Returns:
    - a list of augmented tensors
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    
    if random.random() > 0.5:
        tensors = add_noise(tensors, noise_level=0.01)
    
    if random.random() > 0.5:
        tensors = apply_smoothing(tensors, kernel_size=3)
    
    if random.random() > 0.5:
        tensors = adjust_contrast(tensors, contrast_factor=0.1)
    
    return tensors

class TyphoonDataset(Dataset):
    '''
    A custom dataset class for the Typhoon dataset.
    '''
    def __init__(self, dataset, transform=None, augment=False, seed=42):
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
        self.seed = seed

        # Resize data to 64x64
        self.image_data_resized = np.array([np.array(Image.fromarray(img).resize((64, 64), Image.BICUBIC)) for img in self.image_data])
        self.u10_resized = np.array([np.array(Image.fromarray(u).resize((64, 64), Image.BICUBIC)) for u in self.u10])
        self.v10_resized = np.array([np.array(Image.fromarray(v).resize((64, 64), Image.BICUBIC)) for v in self.v10])
        self.sp_resized = np.array([np.array(Image.fromarray(sp).resize((64, 64), Image.BICUBIC)) for sp in self.sp])
        self.t2m_resized = np.array([np.array(Image.fromarray(t).resize((64, 64), Image.BICUBIC)) for t in self.t2m])

        # calculate min and max values for normalization
        self.sp_min = np.min(self.sp_resized)
        self.sp_max = np.max(self.sp_resized)
        self.t2m_min = np.min(self.t2m_resized)
        self.t2m_max = np.max(self.t2m_resized)

        self.image_min = np.min(self.image_data_resized)
        self.image_max = np.max(self.image_data_resized)
        self.u10_min = np.min(self.u10_resized)
        self.u10_max = np.max(self.u10_resized)
        self.v10_min = np.min(self.v10_resized)
        self.v10_max = np.max(self.v10_resized)

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, index):
        # get the resized image and target data
        img = self.image_data_resized[index]

        # Normalize img to the same range as img_64
        img_normalized = (img - self.image_min) / (self.image_max - self.image_min)
        img_normalized = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)

        # get the resized image (already 64x64)
        img_64 = self.image_data_resized[index]
        img_64 = (img_64 - self.image_min) / (self.image_max - self.image_min)
        img_64 = torch.tensor(img_64, dtype=torch.float32).unsqueeze(0)

        # normalize the target data
        u10 = self.u10_resized[index]
        u10 = (u10 - self.u10_min) / (self.u10_max - self.u10_min)
        u10 = torch.tensor(u10, dtype=torch.float32).unsqueeze(0)

        v10 = self.v10_resized[index]
        v10 = (v10 - self.v10_min) / (self.v10_max - self.v10_min)
        v10 = torch.tensor(v10, dtype=torch.float32).unsqueeze(0)

        sp = self.sp_resized[index]
        sp = (sp - self.sp_min) / (self.sp_max - self.sp_min)
        sp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

        t2m = self.t2m_resized[index]
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

        # copy the image to all channels
        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)

        # data augmentation
        if self.augment:
            tensors = [img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m]
            tensors = conservative_augmentation(tensors, self.seed + index)
            img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = tensors

        return img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m

class CNN(nn.Module):
    '''
    A simple CNN model for the Typhoon dataset.
    '''
    def __init__(self, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.layer1 = self._make_layer(4, 64, stride=2, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, stride=2, dropout_rate=dropout_rate)
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()


    def _make_layer(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        layer = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, dropout_rate),
            ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate)
        )
        return layer   

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upsample1(x4)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.sigmoid(x)
        return x

class ResidualBlock(nn.Module):
    '''
    A residual block for the CNN model.
    '''
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.downsample = nn.Sequential()
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
        out += residual
        out = self.relu(out)
        return out


def train_cnn(train_loader, test_loader, model, optimizer, criterion, device, num_epochs=100):
    '''
    Train the CNN model.
    
    Args:
    - train_loader: the DataLoader for the training dataset
    - test_loader: the DataLoader for the test dataset
    - model: the CNN model
    - optimizer: the optimizer
    - criterion: the loss function
    - device: the device to run the model
    - num_epochs: the number of epochs
    
    Returns:
    - train_losses: a list of training losses
    - test_losses: a list of test losses
    '''
    model.train()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
            targets = [u10, v10, sp, t2m]

            inputs = img_64.to(device)
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = torch.split(outputs, 1, dim=1)
            loss = sum(criterion(output, target) for output, target in zip(outputs, targets))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_epoch_loss:.4f}', flush=True)
        sys.stdout.flush()

        test_loss = evaluate_cnn(test_loader, model, criterion, device)
        test_losses.append(test_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Testing Loss: {test_loss:.4f}', flush=True)
        sys.stdout.flush()

    print('Finished Training')
    torch.save(model.state_dict(), 'cnn_model.pth')
    sys.stdout.flush()
    
    return train_losses, test_losses

def evaluate_cnn(loader, model, criterion, device):
    '''
    Evaluate the CNN model.
    
    Args:
    - loader: the DataLoader for the dataset
    - model: the CNN model
    - criterion: the loss function
    - device: the device to run the model
    
    Returns:
    - the average loss
    '''
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(loader):
            img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
            targets = [u10, v10, sp, t2m]

            inputs = img_64.to(device)
            targets = [t.to(device) for t in targets]

            outputs = model(inputs)
            outputs = torch.split(outputs, 1, dim=1)

            loss = sum(criterion(output, target) for output, target in zip(outputs, targets))
            running_loss += loss.item()

    return running_loss / len(loader)

def load_all_zarr_files(directory):
    '''
    Load all zarr files in the specified directory.
    
    Args:
    - directory: the directory containing the zarr files
    
    Returns:
    - the combined dataset
    '''
    datasets = []
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
    # create the dataset
    set_seed(42)
    base_directory = os.path.join('/vol', 'bitbucket', 'zl1823', 'Typhoon-forecasting','dataset', 'pre_processed_dataset', 'ERA5_without_nan_taiwan')
    combined_dataset = load_all_zarr_files(base_directory)
    
    # split the dataset
    dataset = TyphoonDataset(combined_dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # augment the training dataset
    augmentation_ratio = 0.5
    augmented_size = int(train_size * augmentation_ratio)
    train_indices = train_dataset.indices
    train_combined_data = combined_dataset.isel(time=train_indices)
    augmented_dataset = TyphoonDataset(train_combined_data, augment=True)
    augmented_dataset = random_split(augmented_dataset, [augmented_size, len(augmented_dataset) - augmented_size], generator=torch.Generator().manual_seed(42))[0]
    
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Creating dataloaders...")
    sys.stdout.flush()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=worker_init_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN(dropout_rate=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training CNN Model:")
    train_losses, test_losses = train_cnn(train_loader, test_loader, model, optimizer, criterion, device, num_epochs=50)

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
