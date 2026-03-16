# 3D CNN for Brain Age Prediction

## Overview
* This repository demonstrates the development and deployment of simple 3D Convolutional Neural Networks (CNNs) for predicting brain age from NIfTI brain segmentations. The current examples use FreeSurfer segmentation outputs.
* It includes two distinct CNN architectures: a lightweight version suitable for general computing environments (`LaptopCNN`) and a more powerful, optimized version designed for High-Performance Computing (HPC) environments with GPU acceleration (`HPC_CNN`). Both are pretty simple, mainly for running quick tests - components are kept in a single file for quick demonstration.
* The project also provides a deployment-ready Dockerfile and a prediction script (`predict.py`) for inferencing with trained models.
* One of the main points was testing whether we actually needed many input channels (for many segmentation labels) to achieve SotA performance. Turns out we don't, 1 channel (treating semgnetation as greyscale) will do just fine...Gets ~6 MAE on an 8-88yr dataset.


## Dataset
This project utilizes a dataset of NIfTI MRI brain segmentations, accompanied by age labels.
-   **Preprocessing:** Images undergo normalization, center cropping to a `(182, 182, 182)` shape, and resizing to a target `(140, 140, 140)` resolution before being fed into the CNNs.
-   **Age Normalization:** Age labels are Z-normalized using the mean and standard deviation derived from the training set.
    -   `LaptopCNN` Training Data Normalization (from `training_logs/metadata.json`): Mean=47.60, Std=22.49

* The dataset file directory that looks as follows:
```
.
├── YourDataset1
│   └── dataset1_nipoppy
├── YourDataset2
│   └── dataset2_nipoppy
├── etc...
```
Where the datasetX_nipoppy directory is a Nipoppy-formatted dataset (you only need the derivatives for the purpose of these CNN scripts).

## Model Architectures

### 1. LaptopCNN (`cnn.py`)
A relatively shallow 3D CNN designed for quicker experimentation and environments with limited computational resources.
-   **Layers:** Consists of 5 convolutional blocks, each followed by BatchNorm and LeakyReLU activation, with Dropout layers for regularization.
-   **Output:** A fully connected layer predicts the normalized brain age.
-   **Log Directory:** `training_logs/`

### 2. HPC_CNN (`cnn_hpc.py`)
A beefed-up 3D CNN designed for High-Performance Computing (HPC) environments, leveraging GPU capabilities for more extensive feature learning.
-   **Layers:** A deeper and wider network with 6 convolutional blocks, increased channel depths, and additional dropout layers compared to `LaptopCNN`.
-   **Output:** Features are passed to a more elaborate fully connected head for age prediction.
-   **Optimizations for HPC:** Increased batch size, adjusted learning rate, and higher epoch count to benefit from more powerful hardware.
-   **Log Directory:** `training_logs_hpc/`

## How to Run

### Prerequisites
-   Python 3.9+
-   Docker (for deployment)
-   All Python dependencies listed in `requirements.txt`. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

### Training the LaptopCNN
To train the standard CNN model:
```bash
python cnn.py
```
Results, including the best model weights (`best_model.pth`), training history, and plots, will be saved in the `training_logs/` directory.

### Training the HPC_CNN
To train the HPC-optimized CNN model:
```bash
python cnn_hpc.py
```
Results for this model will be saved in the `training_logs_hpc/` directory.

### Making Predictions
Use the `predict.py` script to make predictions on new NIfTI images. Ensure you have a trained model available in either `training_logs/` or `training_logs_hpc/`.

```bash
# Example using a model trained by cnn.py
python predict.py /path/to/your/nifti_image.nii.gz

# If you want to use the HPC model, you'll need to modify predict.py
# to point to LOGS_DIR = "training_logs_hpc" and MODEL_PATH = "training_logs_hpc/best_model.pth"
# and also update the model class to HPC_CNN.
```
**Note:** For using `predict.py` with the `HPC_CNN`, you would need to manually edit `predict.py` to import `HPC_CNN` and set `LOGS_DIR = "training_logs_hpc"` and `MODEL_PATH = os.path.join(LOGS_DIR, "best_model.pth")` to correctly load the HPC model.

### Docker Deployment
The project includes a `Dockerfile` for easy deployment of the prediction service.

1.  **Build the Docker image:**
    ```bash
    docker build -t brain-age-predictor .
    ```
2.  **Run predictions using the Docker container:**
    Mount your NIfTI file into the container and pass its path to the `predict.py` script.
    ```bash
    docker run --rm -v /path/to/your/nifti_image.nii.gz:/app/input.nii.gz brain-age-predictor /app/input.nii.gz
    ```
    (Ensure that the `best_model.pth` and `metadata.json` are present in `training_logs/` for the Docker image to function correctly with the default `predict.py` configuration.)

