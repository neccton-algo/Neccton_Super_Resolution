# Super Resolution Data Assimilation

This repo contains the Super Resolution Data Assimilation algorithm for the NECCTON project. It contains a notebook illustrating the super-resolution for a specific date and giving some metrics to compare againt the truth and a baseline (bilinear upsampling).

  #### Description : 
A Neural Network allows to go from a low resolution field to a high resolution field to do data assimilation in the HR space, to then go back
in the LR dimension to run a LR model. To do so, the Neural Network computes the residuals it needs to add to a bilinear upsampling of the LR field to get the true HR field.

Once this NN is trained, the full algorithm follows this diagram:
![My Image](./SRDA_diagram.png)

The notebook given only illustrates Step 2. Indeed this work was developed for the specific application of running the TOPAZ data assimilation system, which is based on the coupling of HYCOM-CICE and ECOSMO (for the BGC)
- https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop (HYCOM-CICE)
- git clone https://github.com/pmlmodelling/ersem.git (carbon module)
- https://github.com/nansencenter/TOPAZ_ENKF_BIORAN_v2 (EnKF)

We also include the upsampling and downsampling algorithm. There are specific to the TOPAZ system as well. Their purpose is to upsample , in our case ...
Assuming a model and data assimilation scheme are operational, the scripts that connects the different steps is named Full_inference.sh. It consists in:
- upsamling
- applying the SR operator
- assembling the results (specific step for the binary .ab file format)

Then the script downsample_back_to_TP2.sh is doing the Step 4.

# Usage

To run the super resolution algorithm and compare the result to a bsaline, open Test_ResUnet.ipynb and run all the cells [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AntoineBernigaud/Neccton_Super_Resolution/blob/main/Test_ResUnet.ipynb)

## Data Source

The data required to run the Notebook are:

- High resolution fields (for validation)
- Low resolution fields upsampled to the the HR grid as an input of the neural network
- The High resolution land mask
- The weights of the NN

The HR and LR fields result from a run from the Topaz 5 and Topaz 2 data assimilation system respectively, which relies on a coupled ocean / sea-ice model (HYCOM / CICE) over the Arctic and the North Atlantic sea.
They are available through FTP (details provided in the code). For now, only the Sea Surface Temperature variable is available.

## Baseline

The superresolution with the residual U-Net is compared to bilinear interpolation.

## Metrics

The performance of the super-resolution model is evaluated using the following metrics:

### Bias
Bias indicates the systematic error in predictions, showing whether the model tends to overestimate or underestimate the true values. It is computed as the mean difference between predicted and true values. A bias close to zero suggests balanced predictions, while a significant bias indicates a systematic tendency in the model's predictions.

### Root Mean Squared Error (RMSE) and Peak Signal-to-Noise Ratio (PSNR) 
RMSE quantifies the difference between predicted and true values. It measures the average magnitude of the errors without considering their direction. Lower RMSE values indicate better model performance.

PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. It is simply a scaled log transformation of the MSE, but
is more commonly used in Super Resolution to quantify the quality of the results. The higher the PSNR value, the better the quality of the reconstruction.

### Structural Similarity Index (SSIM)
SSIM measures the similarity between two images. It considers three components: luminance, contrast, and structure. SSIM values range from -1 to 1, with 1 indicating perfect similarity. Higher SSIM values suggest better preservation of image quality.

## List of Dependencies
- Python version: 3.10.5
- TensorFlow version: 2.8.0
- abfile module (available in this repo)
- netCDF4 version: 1.5.8

## Citations and Links
(To be filled)
