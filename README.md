# Semantic Segmentation with DeepLabV3 & Hyperparameter Tuning With Ray Tune

This repository contains a Jupyter notebook for semantic segmentation using the DeepLabV3 model with a ResNet-50 backbone. The notebook demonstrates the process of loading data, training the model, and making predictions. Additionally, Ray Tune is utilized for hyperparameter tuning to optimize the model's performance.

## Prerequisites

- Python 3.6 or later
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- ray[tune]

## Setup

1. Clone the repository:

```sh
git clone https://github.com/yourusername/semantic-segmentation-deeplabv3.git
cd semantic-segmentation-deeplabv3
```
1. Install the required packages:

```shCopy code
pip install torch torchvision numpy matplotlib pillow ray[tune]
```
## Data

- Place your training images in the `train-data/image` directory.
- Place your training labels in the `train-data/label` directory.
- Place your unlabeled images in the `unlabeled-data` directory.
- Place your test images in the `test-data/image` directory.

## Training

The training process includes:

1. Defining a custom dataset class (`ImageDataset`) for loading images and labels.
2. Initializing and customizing a pre-trained DeepLabV3 model.
3. Training the model using the defined dataset and saving the trained model.

To train the model, run the notebook cells related to training.

## Hyperparameter Tuning with Ray Tune

Ray Tune is used to optimize hyperparameters for better model performance. The following steps are included for hyperparameter tuning:

1. **Define the Search Space**: Specify the range of hyperparameters to explore.
2. **Configure the Scheduler and Search Algorithm**: Use Ray Tune's built-in schedulers and search algorithms for efficient hyperparameter tuning.
3. **Run the Tuning**: Execute the tuning process and log the results.
## Evaluation

1. Load the trained model weights.
2. Create a test dataset and DataLoader.
3. Perform inference and save predictions as `.npy` files.
4. Zip the predictions for easy download.

## Results

The training loss is plotted to visualize the training progress. The model predictions are saved and zipped for further analysis.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- Ray Tune