# Doodle Guesser

## Overview

This is a from-scratch Convolutional Neural Network (CNN). It is completely CPU-bound and not optimized in any way. This project exists because I wanted to learn how CNNs work after I built a simple Artifical Neural Network (ANN) and wanted to build a Neural Net (NN) that would be able to preserve spatial relationships. CNNs do this by doing something called 'parameter sharing,' meaning that multiple input parameters are taken into account at the same time.

To understand how this works we look at our input data: an image. An image is a grid where each pixel means something in relation to its neighbors. This relationship is extracted by going to each pixel, grabbing its neighbors (out to some distance, like up to 3 pixels away) which gives use a kind of mini-image of around that pixel. With that mini image (remember just a grid), we multiply the values by a grid of numbers and then add a bias value. Those 2 grids are our weights and biases. The sub-image x weight grid operation is the convolution, and what we did is analogous to a single perceptron, but instead of a vector of weights and a bias, we have a grid of weights and a bias. And per layer, instead of a vector of perceptrons the CNN has a list of filters, or channels.

Just like the perceptron can be scaled up to and ANN we can scale our grid (or kernel) number up as well. Our convolution layer now takes an image, and runs this sub-image x weight kernel convolution operation and outputs another grid of values that we call the feature map. The feature map is just the image after we are left with after running the convolution on the original. For each node in our convolution layer, we have our weights kernel and each node produces its own feature map.

What do we do with the feature maps? We send them along to another layer called a pooling layer. This is a downsampling layer that is similar to a convolution in that it runs a grid over the output image, this time instead of applying the weights and getting a new number, it just takes the largest number. In the end this will produce a smaller image than the input image and the smaller image will keep the most prominent features from the input feature map.

Okay, so we ran our kernels over the image producing new images, then we downsampled those images, whats next? Well, second verse, same as the first. It turns out that you get better results if you just run it through the convolution layer and pooling layer again. I am not sure how many times you are supposed to rinse and repeat, but I have just been doing 1 repeat and getting pretty good results. I do know that it created an 'understanding' heirarchy where early kernels learn edges and simple textures, while deeper ones learn shapes and object parts.

Next? Okay, now is a flatten layer. This takes the image data from the previous layers and just flattens it into a vector, because the next step is just throwing it through an ANN for classification. The previous convolution/pooling stages was to tease the data into a shape that would make the ANN at the end work REALLY well. I should add that the convolution layer applies an activation function, just like the dense layers, like an ANN/MLP.

And that is the feed-forward mechanism for CNNs. The backpropagation is pretty much the same, but backwards. It really is pretty much like the backprop for the dense layers, but instead of a matrix of weights, it's like a matrix of matrices of weights.

The output applies softmax and cross entropy for the loss function. This way the output is normalized so that all outputs fall between 0 and 1 and sum to 1. On the frontend I use argmax to determine the final guess.

## Diagram

```

+----------------------+       +-----------------------+       +-----------------------+
|    Input Image       | --->  |   Conv Layer          | --->  |   Pool Layer          |
| (H x W x Channels)   |       | (filters + activation)|       | (downsample / max)    |
+----------------------+       +-----------------------+       +-----------------------+
                                                                |
                                      .-------------------------|
                                      v
                            +-----------------------+       +-----------------------+
                            |   Conv Layer          | --->  |   Pool Layer          |
                            | (filters + activation)|       | (downsample / max)    |
                            +-----------------------+       +-----------------------+
                                                                |
            .---------------------------------------------------|
            v
+-----------------------+       +-----------------------+       +-----------------------+
|   Flatten Layer       | --->  |  Dense / ANN Layers   | --->  |   Output Layer        |
| (feature maps → vec)  |       | (MLP + activation)    |       | (Softmax + CrossEnt)  |
+-----------------------+       +-----------------------+       +-----------------------+
                                                                |
                                            .-------------------|
                                            v
                                +-------------------------+
                                | Class Probabilities     |
                                | argmax -> Final Guess   |
                                +-------------------------+

```

## Running the Frontend

First download the repository and run `npm install` at the root, This will install all of the node modules needed. Then to run the application:

```bash
npm run run:frontend
```

This will start the vite application, by default the applicaition will be accessible via `http://localhost:5173/`.

The frontend application will load the model file and training manifest. The `frontend/src/main.ts` is where the model and manifest files are set, and are loaded from the `frontend/public` folder.

## Training a Model

This command will train a model to detect cat, butterfly and rainbow doodles, using 2000 training images and 500 testing images. It will output the file `doodle-guesser-3.json` to the `backend/` folder. This is the model file, and you can find the training file in the `backend/.cache/public` folder under the name `doodle-manifest.json`. Once the training has completed you can move the model and manifest into the `frontend/public` folder and update the `frontend/src/main.ts` file to point to the desired model and manifest files.

```bash
npm run train --workspace=@doodle/backend -- --class-list="src/categories-3.txt" --train-count=2000 --test-count=500 --model-output="doodle-guesser-3.json"
```

CLI Usage

```bash
Usage: doodle-trainer [optional_arg]
Builds data set and trains doodle detector CNN

Options:
  --help                    Prints out this help message
  --class-list="<file>"     Text file list of doodle types, newline delimited (default: src/categories-3.txt)
  --train-count=<number>    Number of doodles to train on, per doodle, per epoch
  --test-count=<number>     Number of doodles to test against per doodle
  --model-output="<file>"   Path to the final model file output

Examples:
  npm run train --workspace=@doodle/backend -- --class-list="src/categories-3.txt" --model-output="doodle-detector.json"
```
