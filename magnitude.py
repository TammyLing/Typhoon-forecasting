import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xarray as xr
import random
import sys
import logging
import argparse
from ema_pytorch import EMA
from torchvision import transforms as T
from CDDPM.models.model import Palette
from CDDPM.models.network import Network
from cnn import CNN
from SENet.senet import SENet
from DDPM.diffusion import GaussianDiffusion, Unet
from CDDPM.core.logger import VisualWriter, InfoLogger
import CDDPM.core.praser as Praser
import CDDPM.core.util as Util
from CDDPM.data import define_dataloader
from CDDPM.models import create_model, define_network, define_loss, define_metric

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def denormalize(tensor, min_val, max_val):
    '''
    Denormalize a tensor using the given min and max values.
    '''
    return tensor * (max_val - min_val) + min_val

class TyphoonDataset(Dataset):
    '''
    Dataset class for the Typhoon dataset.
    '''
    def __init__(self, dataset, denorm_dataset, times, transform=None, augment=False, seed=42):
        super().__init__()
        self.dataset = dataset
        self.denorm_dataset = denorm_dataset
        self.times = times

        self.image_data = self.dataset['image_data'].values
        self.u10 = self.dataset['u10'].values
        self.v10 = self.dataset['v10'].values
        self.sp = self.dataset['sp'].values
        self.t2m = self.dataset['t2m'].values

        self.transform = transform
        self.augment = augment
        self.seed = seed

        self.image_data_resized = np.array([np.array(Image.fromarray(img).resize((64, 64), Image.BICUBIC)) for img in self.image_data])
        self.u10_resized = np.array([np.array(Image.fromarray(u).resize((64, 64), Image.BICUBIC)) for u in self.u10])
        self.v10_resized = np.array([np.array(Image.fromarray(v).resize((64, 64), Image.BICUBIC)) for v in self.v10])
        self.sp_resized = np.array([np.array(Image.fromarray(sp).resize((64, 64), Image.BICUBIC)) for sp in self.sp])
        self.t2m_resized = np.array([np.array(Image.fromarray(t).resize((64, 64), Image.BICUBIC)) for t in self.t2m])

        # Calculate min and max values for normalization
        self.sp_min = self.denorm_dataset.sp.min().item()
        self.sp_max = self.denorm_dataset.sp.max().item()
        self.t2m_min = self.denorm_dataset.t2m.min().item()
        self.t2m_max = self.denorm_dataset.t2m.max().item()

        self.image_min = self.image_data.min().item()
        self.image_max = self.image_data.max().item()
        self.u10_min = self.denorm_dataset.u10.min().item()
        self.u10_max = self.denorm_dataset.u10.max().item()
        self.v10_min = self.denorm_dataset.v10.min().item()
        self.v10_max = self.denorm_dataset.v10.max().item()

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, index):
        img = self.image_data_resized[index]
        img_normalized = (img - self.image_min) / (self.image_max - self.image_min)
        img_normalized = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)

        img_64 = self.image_data_resized[index]
        img_64 = (img_64 - self.image_min) / (self.image_max - self.image_min)
        img_64 = torch.tensor(img_64, dtype=torch.float32).unsqueeze(0)

        u10 = self.u10_resized[index].astype(np.float32)
        u10 = torch.tensor(u10, dtype=torch.float32).unsqueeze(0)

        v10 = self.v10_resized[index].astype(np.float32)
        v10 = torch.tensor(v10, dtype=torch.float32).unsqueeze(0)

        sp = self.sp_resized[index].astype(np.float32)
        sp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

        t2m = self.t2m_resized[index].astype(np.float32)
        t2m = torch.tensor(t2m, dtype=torch.float32).unsqueeze(0)

        original_u10 = self.u10[index].astype(np.float32)
        original_v10 = self.v10[index].astype(np.float32)
        original_sp = self.sp[index].astype(np.float32)
        original_t2m = self.t2m[index].astype(np.float32)

        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)
        time = str(self.times[index])
        if '.' in time:
            time = time.split('.')[0]

        return img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m, time

def load_all_zarr_files(directory):
    '''
    Load all zarr files in the specified directory and concatenate them along the time dimension.
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
        raise ValueError("No zarr files found in the specified directory.")
    combined_dataset = xr.concat(datasets, dim="time")
    return combined_dataset

def set_seed(seed):
    '''
    Set the seed for reproducibility.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cnn_model(model, model_path, device):
    '''
    Load a CNN model from the specified path.
    '''
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

def load_diffusion_model(model, model_path, device):
    '''
    Load a Diffusion model from the specified path.
    '''
    state_dict = torch.load(model_path, map_location=device)
    model_state_dict = state_dict['model']
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    if 'ema' in state_dict:
        print("Loading EMA...")
        ema_model = EMA(model, beta=0.995)
        ema_model.load_state_dict(state_dict['ema'])
        ema_model.to(device)
        model.ema = ema_model
    else:
        print("EMA not found in state_dict.")
    return model

def load_con_diffusion_model(model, model_path, device):
    '''
    Load a Conditional Diffusion model from the specified path.
    '''
    state_dict = torch.load(model_path, map_location=device)
    model.netG.load_state_dict(state_dict, strict=False)
    model.netG.to(device)
    model.netG.set_new_noise_schedule(device=device, phase='test')
    print(f"num_timesteps after setting noise schedule: {model.netG.num_timesteps}")
    model.netG.num_timesteps = model.netG.num_timesteps if hasattr(model.netG, 'num_timesteps') else 1000
    if 'ema' in state_dict:
        print("Loading EMA...")
        ema_model = EMA(model, beta=0.995)
        ema_model.load_state_dict(state_dict['ema'])
        ema_model.to(device)
        model.ema = ema_model
    else:
        print("EMA not found in state_dict.")
    return model

def plot_comparisons(test_loader, models, device, dataset):
    '''
    Generate predictions and comparisons for the given models.
    '''
    results_folder = "muifa_comparison_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    for batch_idx, data in enumerate(test_loader):
        _, img_64, u10, v10, _, _, _, _, _, _, time = data
        inputs = img_64.to(device)
        
        for i in range(inputs.size(0)): 
            fig, axs = plt.subplots(1, len(models) + 2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # get the lon and lat
            lon_start, lon_end = 116.0794, 126.0794
            lat_start, lat_end = 18.9037, 28.9037
            lon = np.linspace(lon_start, lon_end, 64)
            lat = np.linspace(lat_start, lat_end, 64)
            extent = [lon_start, lon_end, lat_start, lat_end]

            # Plot input image
            axs[0].imshow(img_64[i][0].cpu().numpy().squeeze(), cmap='gray', extent=extent, transform=ccrs.PlateCarree())
            axs[0].set_extent(extent)
            axs[0].coastlines()
            axs[0].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', alpha=0.5)
            axs[0].set_title('Input Image')

            # Add time to the input image plot
            axs[0].text(0.5, -0.1, f'Time: {time[i]}', ha='center', va='top', transform=axs[0].transAxes)

            all_magnitudes = []
            for idx, (model_name, model) in enumerate(models.items()):
                model_class_name = model.__class__.__name__

                if model_class_name in ['GaussianDiffusion', 'Palette']:
                    if model_class_name == 'GaussianDiffusion':
                        model.ema.ema_model.eval()
                        with torch.inference_mode():
                            t = torch.randint(0, 1000, (inputs.size(0),), device=device).long()
                            noise = torch.randn_like(inputs)
                            noisy_input = model.q_sample(inputs, t, noise)
                            pred_img = noisy_input.clone()
                            for time_step in reversed(range(model.num_timesteps)):
                                pred_img, _ = model.p_sample(pred_img, time_step)
                            outputs = pred_img.split(1, dim=1)
                    elif model_class_name == 'Palette':
                        if hasattr(model.netG, 'module'):
                            netG = model.netG.module
                        else:
                            netG = model.netG
                        netG.eval()
                        with torch.inference_mode():
                            cond_image = inputs
                            outputs, _ = netG.restoration(cond_image, sample_num=8)
                            outputs = outputs.split(1, dim=1)
                elif model_class_name in ['ConditionCNN', 'SENet']:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(inputs)
                        outputs = outputs.split(1, dim=1)
                
                output_u10 = outputs[0][i].cpu().numpy().squeeze()
                output_v10 = outputs[1][i].cpu().numpy().squeeze()

                # Denormalize the output data
                output_u10_denorm = denormalize(torch.tensor(output_u10), dataset.u10_min, dataset.u10_max).numpy()
                output_v10_denorm = denormalize(torch.tensor(output_v10), dataset.v10_min, dataset.v10_max).numpy()

                # Calculate magnitude
                output_magnitude = np.sqrt(output_u10_denorm**2 + output_v10_denorm**2)
                all_magnitudes.append(output_magnitude)

            # Calculate and plot target magnitude
            target_u10 = u10[i].cpu().numpy().squeeze()
            target_v10 = v10[i].cpu().numpy().squeeze()
            target_magnitude = np.sqrt(target_u10**2 + target_v10**2)
            all_magnitudes.append(target_magnitude)

            # Determine global min and max across all magnitudes
            vmin = min(map(np.min, all_magnitudes))
            vmax = max(map(np.max, all_magnitudes))

            # Plot model outputs with unified colorbar range
            for idx, output_magnitude in enumerate(all_magnitudes[:-1]):
                cs = axs[idx + 1].contourf(lon, lat, output_magnitude, cmap='coolwarm', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), alpha=0.9)
                axs[idx + 1].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
                axs[idx + 1].coastlines()
                axs[idx + 1].set_title(f'{list(models.keys())[idx]} Magnitude')

            # Plot target magnitude with unified colorbar range
            cs = axs[-1].contourf(lon, lat, target_magnitude, cmap='coolwarm', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), alpha=0.9)
            axs[-1].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
            axs[-1].coastlines()
            axs[-1].set_title('Target Magnitude')

            # Add colorbar to the right of all plots
            cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('Magnitude')

            plt.savefig(os.path.join(results_folder, f"comparison_{batch_idx}_{i}.png"))
            plt.close()


def main():
    set_seed(42)
    # Load dataset
    print("Loading dataset...")
    sys.stdout.flush()
    base_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/muifa'
    denorm_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/dataset/pre_processed_dataset/ERA5_without_nan_taiwan'
    combined_dataset = load_all_zarr_files(base_directory)
    denorm_dataset = load_all_zarr_files(denorm_directory)

    # Create dataset and dataloader
    print("Creating datasets...")
    sys.stdout.flush()
    dataset = TyphoonDataset(combined_dataset, denorm_dataset, combined_dataset['time'].values)
    # denorm_dataset = TyphoonDataset(denorm_dataset, denorm_dataset, denorm_dataset['time'].values)

    print("Creating dataloaders...")
    sys.stdout.flush()
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader([], batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ''' parser configs '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/config/typhoon.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    trained_models = {
        'CNN': CNN(),
        'SENet': SENet(),
        'DDPM': GaussianDiffusion(Unet(dim=64, channels=4), image_size=(64, 64)),
        'CDDPM': create_model(
            opt=opt,
            networks=networks,
            phase_loader=train_loader,
            test_loader=test_loader,
            losses=losses,
            metrics=metrics,
            logger=phase_logger,
            writer=phase_writer
        )
    }

    model_files = {
        'CNN': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/cnn_normal_model.pth',
        'SENet': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/cnn_se_normal_model.pth',
        'DDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/results/model-30.pt',
        'CDDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/experiments/train_typhoon_forecasting_240803_211802/checkpoint/3350_Network.pth',
    }

    # Load models
    trained_models['CNN'] = load_cnn_model(trained_models['CNN'], model_files['CNN'], device)
    trained_models['SENet'] = load_cnn_model(trained_models['SENet'], model_files['SENet'], device)
    trained_models['DDPM'] = load_diffusion_model(trained_models['DDPM'], model_files['DDPM'], device)
    trained_models['CDDPM'] = load_con_diffusion_model(trained_models['CDDPM'], model_files['CDDPM'], device)

    print("Generating predictions and comparisons...")
    sys.stdout.flush()
    plot_comparisons(test_loader, trained_models, device, dataset)

if __name__ == '__main__':
    main()
