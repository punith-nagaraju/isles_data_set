# ISLES'22 Ensemble Segmentation

This project implements ensemble learning techniques for medical image segmentation using the ISLES'22 dataset. The goal is to enhance segmentation performance by combining multiple models through various ensemble methods.

## Project Structure

```
isles22-ensemble-segmentation
├── src
│   ├── dataset.py          # Custom PyTorch Dataset class for loading ISLES'22 data
│   ├── augmentations.py     # Data augmentation techniques using albumentations
│   ├── model.py             # Base segmentation models using segmentation-models-pytorch
│   ├── train.py             # Training loop for base segmentation models
│   ├── ensemble.py          # Ensemble techniques for model predictions
│   ├── predict.py           # Prediction process using the trained ensemble model
│   └── utils.py             # Utility functions for model loading and evaluation metrics
├── data                     # Directory for placing the ISLES'22 dataset
│   └── (place ISLES'22 dataset here)
├── requirements.txt         # Required Python packages for the project
└── README.md                # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd isles22-ensemble-segmentation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place the ISLES'22 dataset in the `data` directory. Ensure that the dataset contains the necessary NIfTI files for DWI, ADC, and FLAIR modalities.

## Usage Guidelines

- To train the base segmentation models, run:
  ```
  python src/train.py
  ```

- To perform predictions using the trained ensemble model, execute:
  ```
  python src/predict.py
  ```

## Implemented Methods and Models

- **Dataset Class**: Custom PyTorch Dataset for loading and preprocessing the ISLES'22 data.
- **Data Augmentation**: Various augmentation techniques to enhance model robustness.
- **Segmentation Models**: Implementations of U-Net and its variants for segmentation tasks.
- **Training Loop**: Includes optimizer setup, loss function definitions, and training steps.
- **Ensemble Techniques**: Simple averaging, weighted averaging, and stacking methods for combining model predictions.
- **Prediction Handling**: Processes input data and generates final segmentation masks based on ensemble predictions.
- **Utility Functions**: For model loading, evaluation metrics, and result visualization.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.