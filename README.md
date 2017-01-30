# CarND-Term Project 3

## Introduction

In this project, we are required to train a model that can drive a car in a simulator.

## Data prepocessing

### Steering angle shift

Because in the autonomous, the app receive an image then send the steering data. During several experienment, I find that
the vehicule sometimes turned too late.
So In the training, I decide modify the data
A image's correspond steering = 1/2 * (this moment + next moment (0.1s later))

### Delete too slow data

If the speed is too low, the angle doesn't make sense, so I delete them

### Reinforce big turned

The training data mainly composed by straight driving data. To train more on turning, I duplicate the training data for 
steering_angle > 0.2, 5 times.

### Left and Right camera

I add(minus) 0.045 steering angle for left(right) camera 

### validation set

data is split as 4:1 for training and validation

## Architecture

   160 x 320 x 3
-> BatchNormalization

-> Convolution2D (kernal=11 stride=2) -> 75 x 155 x 40 -> Relu
-> MaxPolling2D (stride = 2)          -> 37 x 77 x 40

-> Convolution2D (kernal=7 stride=2)  -> 16 x 36 x 60  -> Relu
-> MaxPolling2D(stride = 2)           -> 8 x 18 x 60

-> Convolution2D (kernal=3)           -> 6 x 16 x 80   -> Relu
-> Convolution2D (kernal=(3, 1))      -> 4 x 16 x 110  -> Relu 

-> Flatten 7040
-> Dropout(0.5)

-> Dense 1000 -> Relu
-> Dense 200 -> Relu
-> Dense 50 -> Relu
-> Dense 10 -> Relu
-> Dense 1

## Training

The total data set contains 100 000 pictures.

The following data is an example for one of the training.

Epoch 1/20
116424/116367 [==============================] - 406s - loss: 0.0092 - val_loss: 0.0080                                            0.0065
Epoch 2/20
116424/116367 [==============================] - 412s - loss: 0.0069 - val_loss: 0.0062
Epoch 3/20
116424/116367 [==============================] - 409s - loss: 0.0060 - val_loss: 0.0054
Epoch 4/20
116424/116367 [==============================] - 406s - loss: 0.0053 - val_loss: 0.0052
Epoch 5/20
116424/116367 [==============================] - 406s - loss: 0.0048 - val_loss: 0.0049
Epoch 6/20
116424/116367 [==============================] - 401s - loss: 0.0043 - val_loss: 0.0045
Epoch 7/20
116424/116367 [==============================] - 403s - loss: 0.0039 - val_loss: 0.0041
Epoch 8/20
116424/116367 [==============================] - 392s - loss: 0.0034 - val_loss: 0.0038
Epoch 9/20
116424/116367 [==============================] - 392s - loss: 0.0031 - val_loss: 0.0034
Epoch 10/20
116424/116367 [==============================] - 397s - loss: 0.0028 - val_loss: 0.0033
Epoch 11/20
116424/116367 [==============================] - 392s - loss: 0.0026 - val_loss: 0.0030
Epoch 12/20
116424/116367 [==============================] - 389s - loss: 0.0024 - val_loss: 0.0030
Epoch 13/20
116424/116367 [==============================] - 392s - loss: 0.0022 - val_loss: 0.0028
Epoch 14/20
116424/116367 [==============================] - 393s - loss: 0.0021 - val_loss: 0.0027
Epoch 15/20
116424/116367 [==============================] - 397s - loss: 0.0018 - val_loss: 0.0026
Epoch 16/20
116424/116367 [==============================] - 392s - loss: 0.0018 - val_loss: 0.0024
Epoch 17/20
116424/116367 [==============================] - 398s - loss: 0.0016 - val_loss: 0.0024
Epoch 18/20
116424/116367 [==============================] - 420s - loss: 0.0016 - val_loss: 0.0021
Epoch 19/20
116424/116367 [==============================] - 419s - loss: 0.0015 - val_loss: 0.0022
Epoch 20/20
116424/116367 [==============================] - 402s - loss: 0.0014 - val_loss: 0.0021

In this training, the model is a little bit overfitted. Thus for this experienment, the best epoch is around 15.

As a comparison, the overall Steering MSE:  0.029642653027202204, so the model perform quite well on this number.

However, all models trained are not stable and perform not well in test. 

## Other Try

I also tried transfer learning. Build several model based on Inception_V3, VGG19...
All works not well...
Worked about 100 hours on it, more than 20 models...