# Neccton_Super_Resolution

Super Resolution for the NECCTON project

## Data Source

TOPAZ sea surface temperature and neural network weights are available through FTP (details provided in the code).

## Baseline

The superresolution with the residual U-Net is compared to bilinear interpolation.

## Metrics

The performance of the super-resolution model is evaluated using the following metrics:

### Root Mean Squared Error (RMSE)
RMSE quantifies the difference between predicted and true values. It measures the average magnitude of the errors without considering their direction. Lower RMSE values indicate better model performance.

### Structural Similarity Index (SSIM)
SSIM measures the similarity between two images. It considers three components: luminance, contrast, and structure. SSIM values range from -1 to 1, with 1 indicating perfect similarity. Higher SSIM values suggest better preservation of image quality.

### Bias
Bias indicates the systematic error in predictions, showing whether the model tends to overestimate or underestimate the true values. It is computed as the mean difference between predicted and true values. A bias close to zero suggests balanced predictions, while a significant bias indicates a systematic tendency in the model's predictions.

These metrics provide insights into the performance of the super-resolution model, helping to assess its accuracy and reliability.

## List of Dependencies
- Python
- TensorFlow

## Citations and Links
(To be filled)
