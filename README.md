# Potato Disease Classification Model

Classifies potato leaf images as healthy or diseased (early blight / late blight) with a convolutional neural network built in TensorFlow/Keras.

## Dataset

Potato leaf images in `Dataset/` (also available as `Dataset.zip`), in three class folders:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

Images are loaded at 256x256 with batch size 32, split into train, validation, and test sets, then cached, shuffled, prefetched, resized, normalized, and augmented (random flips/rotations) before training.

## Model

A Sequential CNN:

- Six Conv2D + MaxPooling2D blocks (32 filters, then 64 filters, 3x3 kernels, ReLU)
- Flatten, Dense(64, relu), Dense(3, softmax)
- **Training:** Adam optimizer, sparse categorical cross-entropy loss, 20 epochs

## Results

As evaluated in the notebook:

- **Test accuracy: 96.48%** (loss 0.094) on the held-out test set
- **Best validation accuracy: 97.92%** during training

## Project structure

```
potato-disease-classification.ipynb   # data loading, augmentation, training, evaluation
Dataset/                              # potato leaf images by class
Dataset.zip                           # the same dataset as a zip archive
```

## How to run

1. Install the dependencies: TensorFlow, Matplotlib.
2. Open `potato-disease-classification.ipynb` in Jupyter and run the cells in order.
3. Note: the notebook was run on Kaggle and loads the data from `/kaggle/input/dataset/Dataset` — point that path at the local `Dataset/` folder to run it elsewhere.
