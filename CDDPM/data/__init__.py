from functools import partial
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import os
import sys
import xarray as xr
import random
import torch
from data.dataset import TyphoonDataset

def set_seed(seed):
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

def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    phase_dataset, test_dataset = define_dataset(logger, opt)

    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False})

    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    if opt['global_rank'] == 0 and test_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args', {}))
        test_dataloader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, **dataloader_args)
    else:
        test_dataloader = None

    return dataloader, test_dataloader

def load_all_zarr_files(directory):
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

def define_dataset(logger, opt):
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    dataset_directory = dataset_opt['args']['data_root']
    combined_dataset = load_all_zarr_files(dataset_directory)
    phase_dataset = TyphoonDataset(combined_dataset)
    train_size = int(0.8 * len(phase_dataset))
    val_size = len(phase_dataset) - train_size
    set_seed(42)
    train_dataset, test_dataset = random_split(phase_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    augmentation_ratio = 0.5
    augmented_size = int(train_size * augmentation_ratio)
    train_indices = train_dataset.indices
    train_combined_data = combined_dataset.isel(time=train_indices)
    augmented_dataset = TyphoonDataset(train_combined_data, augment=True)
    augmented_dataset = random_split(augmented_dataset, [augmented_size, len(augmented_dataset) - augmented_size], generator=torch.Generator().manual_seed(42))[0]

    phase_dataset = ConcatDataset([train_dataset, augmented_dataset])

    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = dataloder_opt.get('validation_split', 0)

    logger.info('Dataset for {} have {} samples.'.format('train', len(phase_dataset)))
    logger.info('Dataset for {} have {} samples.'.format('test', len(test_dataset)))
    print(f"Training dataset size: {len(phase_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return phase_dataset, test_dataset

def subset_split(dataset, lengths, generator):
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
