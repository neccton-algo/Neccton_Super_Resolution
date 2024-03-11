# Neccton_Super_Resolution

Super Resolution for the NECCTON project

## Data Source

TOPAZ sea surface temperature and neural network weights are available through FTP (details provided in the code).

## Baseline

The superresolution with the residual U-Net is compared to bilinear interpolation.

## Metrics

The performance of the super-resolution model is evaluated using the following metrics:

### Root Mean Squared Error (RMSE)
RMSE quantifies the difference between predicted and true values. It is calculated using the following formula:
\[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \]
Where:
- \( n \) is the number of samples.
- \( y_i \) is the true value.
- \( \hat{y}_i \) is the predicted value.

### Structural Similarity Index (SSIM)
SSIM measures the similarity between two images. It ranges from -1 to 1, with 1 indicating perfect similarity. SSIM is computed across multiple windows of the images and then averaged.
\[ \text{SSIM}(x, y) = \frac{{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}}{{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}} \]
Where:
- \( \mu_x \) and \( \mu_y \) are the mean values of \( x \) and \( y \) respectively.
- \( \sigma_x^2 \) and \( \sigma_y^2 \) are the variances of \( x \) and \( y \) respectively.
- \( \sigma_{xy} \) is the covariance of \( x \) and \( y \).
- \( C_1 \) and \( C_2 \) are small constants to stabilize the division with weak denominator.

### Bias
Bias indicates the systematic error in predictions, showing whether the model tends to overestimate or underestimate the true values. It is computed as the mean difference between predicted and true values.

These metrics provide insights into the performance of the super-resolution model, helping to assess its accuracy and reliability.

## List of Dependencies
- Python
- TensorFlow

## Citations and Links
(To be filled)

Feel free to update this section as necessary.
