# Typhoon Forecasting Project

## Overview
This project focuses on predicting typhoon patterns using various deep learning models. It involves downloading and processing datasets, selecting typhoons passing through specific regions, aligning different datasets, and training models to forecast typhoon behavior. The project includes training four different models, including DDPM and CDDPM, with some parts of the model architecture adapted from other open-source projects.

## Requirements
To set up the environment and install the necessary dependencies, run the following commands:

```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Download ERA5 and DT Data
- Download ERA5 data using the Jupyter notebook provided: `dataset/ERA5 - Data Download.ipynb`.
- For DT data, download the relevant typhoon datasets from the official DT website: http://agora.ex.nii.ac.jp/digital-typhoon/index.html.en.

### 2. Select Typhoons in Specific Regions
- Run the script `select_typhoons.py` to filter typhoons that have passed through a specified region:

    ```bash
    python dataset/select_typhoons.py
    ```

### 3. Crop the Region
- Use `fix_region.py` to crop the typhoon data to the region of interest:

    ```bash
    python dataset/fix_region.py
    ```

### 4. Align ERA5 and DT Data
- To align the ERA5 dataset with the processed DT data, run `transfer_tarr.py`:

    ```bash
    python dataset/transfer_tarr.py
    ```

## Model Training

This project involves training four models to forecast typhoons. 

The **DDPM model** is based on the architecture from the [lucidrains' denoising-diffusion-pytorch repository](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main?tab=readme-ov-file).

The **CDDPM model** is based on the framework from the [Palette: Image-to-Image Diffusion Models repository](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

## Model Evaluation

### 1. Performance Matrix and pixel difference
To evaluate the model performance and pixel difference, run the `evaluation.py` script to obtain the performance matrix.

### 2. Magnitude Calculation
To calculate the predicted wind magnitude using predicted `u10` and `v10`, run the `magnitude.py` script.

### 3. Case study

#### 1. Download Typhoon Data
- Download the CSV file containing typhoon data from the IBTrACS website: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/.

#### 2. Generate Prediction Animation
- Run the `muifa_compare.py` script to generate an animation comparing the predicted and actual atmospheric variables.

## Acknowledgments

- The **DDPM model** implementation is adapted from the repository [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
- The **CDDPM model** is based on the architecture from the [Palette repository](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for more details.


