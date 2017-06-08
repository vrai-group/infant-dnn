# Infant segmentation with Deep Neural Networks

The aim of the project is to extract the shape of infants from a set of depth images.

First of all it was created a dataset of labelled images taken from several recordings.

Secondly the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)  was appositely configured to train itself on the dataset previous mentioned.

Lastly the U-Net trained model should predict, getting new depth images as input, the shape of infants. 

## Prerequisites
This project was developed on Python 3.6 and needs the following libraries:
* numpy
* scipy
* scikit-image
* scikit-learn
* Theano/Tensorflow and Keras

Download link: [Libraries](http://www.lfd.uci.edu/~gohlke/pythonlibs/)


## How to use

### Prepare the data

In order to obtain `*.npy` files from `data.py` script, you have to create the following directory structure. The `positive` and `partial` folders contain 8-bit images, 16-bit images and mask images.

```
-raw
 |
 ---- train
 |    |
 |    ---- positive
 |               |
 |                 ---- image.png
 |    ---- partial
 |               |
 |                 ---- image_partial.png
```

Now run ```python data.py```.
This script randomly splits all images in `training` and `testing` set through the variable `test_percentage`, that represents the relationship between test and total images.
Running this script will create train and test images and save them to **.npy** files.

### Training the model
Once the `npy` files was created, `train.py` must be runned in order to create the model of the U-net and to train the neural network.
The specific parameters of the U-net (epochs, batch_size) can be configured before the running.
After that it will be created `weights.h5` file (containing the weighted values that synthesize the trained model) that will we be used in the `model.predict` function in order to produce the masks associated with `*_test.npy` images.
A value named "accuracy" will show the success percentage of the prediction.

### Testing the model
Once the U-net was trained, you can move `*_train.npy` and `weights.h5` files into the `trained model` folder in order to achieve a faster prediction using the script `test.py` without training the model again.
The previous described `test.py` script get as input the `*_test.npy` files created by `data.py` and the `*_train.npy`files moved in the mentioned folder.

```
-trained model
 |
 ---- *_train.npy
 |    |
 ---- weights.h5
```

## Contributing

Please read [Marko Jocic repository](https://github.com/jocicmarko/ultrasound-nerve-segmentation) for more detailed information about the U-net architecture. 



## Authors

* **Matteo Sartori** - [GitHub](https://github.com/matteosartori)
* **Luca Virgili**  - [GitHub](https://github.com/lucav48)
* **Jacopo Zincarini**  - [GitHub](https://github.com/jacopozincarini)


