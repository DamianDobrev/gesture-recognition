# gesture-recognition
Gesture recognition software for controlling a drone

## Description
This project consists of tools for data collection, training and real time prediction of 
hand gestures. All images are processed before feeding them into the model, as well as
before trying to predict their class. This is done to help with the accuracy of the model.
Image processing is done in multiple ways, and can vary more, but it is important that 
while predicting, the model is fed with data processed in the same way as it is has been
trained initially.

The software provides real-time prediction of the image captured by the computer camera.
Each frame is processed, then fed into the predictor. Details on processing below.

There are a total of 9 classes - ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock'].

## Image Preprocessing
Since the output is continues (video), to predict values continuously, each frame is processed
and predicted independently. Here some of the image processing algorithms are described:

#### Hand extraction by skin color HSV thresholding
- The raw frame BGR frame is loaded
- Image is converted into HSV
- The skin color is extracted based on HSV threshold range
- A binary masked is created for each px that is found to be skin color
- To reduce noise, the largest connected component within the binary mask is found, 
    everything else is discarded
- The binary mask may contain some holes due to different shades of the hand colors 
    not being captured properly by the thresholding. To reduce this, a series of erosions
    and dilations are applied, as well as some algorithms for filling all holes
- The binary mask is applied to the original image, where the `1`s from the binary mask
    are the original image colors and the `0`s from the binary mask are `0`s in the 
    resulting image as well.
- The resulting image is the skin on black background.

#### Bounding boxes and crop
- Having the binary mask with reduced noise, the rectangular bounding box around the binary 
    mask is computed.
- The rectangular bounding box is then used to calculate a square bounding box around the 
    binary mask. This ensures the hand is always in the middle of the frame and does not
    depend on how close it is to the camera. 
    
    For example, if the hand appears to be 60x100px
    on the original image, the square bounding box is 100x100px around the hand. Then if 
    that bounding box is scaled up to 200x200px, the hand is effectively 120x200px. 
    
    If initially the same gesture is performed further from the 
    camera and the size of it is 30x50px, then the bounding box is 50x50, and after scaling
    to 200x200px, the size of the hand is now 120x200px, which is the same size as in the
    first case, although the distance is twice as big.
    
Since this operation is performed both during training and during prediction, the spatial 
constraint regarding translation and size is almost fully neglected, which helps improve the
performance of the network.

#### Color processing
To speed up training as well as remove the color constraint, the image is further processed
before prediction or training. 
- The image with extracted skin on black background is converted to grayscale, hence the 
    shape is now `(x,y,1)`.
- The histogram is equalized using cv2's `equalizeHist` method, this bumps up or down the
    contrast, which increases the strength of the features.
- The image is rescaled to smaller size, e.g. 50x50px.
- Image is fed to the model 


## App Structure
The app consists of 4 main programs - Calibrator, Data Collector, Model Trainer, Predictor.

#### Calibrator
The calibrator is used to find a good HSV threshold to extract skin colors. It is run before
predicting and data collection. It is not run before training, because the training program
should use the same hsv thresholds as the data collector used during the collection of data.
This value should be given in the `training/hsv_ranges.csv` file, as the first two values of 
the CSV, first line for lower, second line for upper range.

If the calibrator is "skipped", it immediately returns default values, which are fetched 
from `training/hsv_ranges.csv`.
 
#### Data Collector
This program can be used to extract data images. It runs a loop, and saves images every
N milliseconds. The folder where the images are saved is created in `config.CONFIG['path_to_raw']`,
and that folder is called with the name specified in `config.CONFIG['class_to_collect_data']`.
This class should be a numerical value, equal to the `idx+1` in the list 
`config.CONFIG['classes']`. E.g., if `classes: ['cat', 'dog', 'shark']`, to collect images
for the "dog" class, the `class_to_collect_data` in CONFIG should be set to `2`.

#### Model Trainer
TODO
This will save in `./_results/timestamp-of-finish` the result

#### Predictor
TODO


## HPC
I have tried using HPC to train, but it seems I don't ned it. For future reference, to run
on hpc, modify the `deployhpc.sh` file to match the correct ssh destination,
and modify the `exclude-list.txt` to exclude all folders that are not needed for the
trainer to train.

This may not be needed, training times are small due to the images being
only one channel, as well as their size and number (below 100x100px, less than 1000 entrties 
per class).

Training time on my MacBook PRO 2017 13" - around 20-30 min.