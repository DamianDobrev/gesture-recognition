# gesture-recognition
Gesture recognition software for controlling a drone

## Description
This project consists of tools for data collection, training and real time prediction of 
hand gestures. All images are processed before being fed into the model, as well as
before predicting their class. This is done to help improve the accuracy of the model.
Image processing is done in multiple ways, and can be expanded, but it is important that 
while predicting, the model is fed with data processed in the same way as it is has been
trained initially. This works automatically.

The software provides real-time prediction of the image captured by the computer camera.
Each frame is processed, then fed into the predictor. Details on processing below.

The classes I am using are 9 in total: ['stop', 'palm', 'right', 'left', 'hover', 'updown', 
'fist', 'peace', 'rock'], but those can of course be changed depending on the needs.

## Image Processing
Since the output of the camera is continues (video), to predict values continuously, each frame is processed
and predicted independently.

#### Hand extraction by skin color HSV thresholding
To improve the results, the hand is extracted from the background using lower and upper range
HSV thresholding. The following roughly describes the algorithm:
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
This value should be given in the `/hsv_ranges.csv` file, as the first two values of 
the CSV, first line for lower, second line for upper range.

If the calibrator is "skipped", it immediately returns default values, which are fetched 
from `/hsv_ranges.csv`.
 
#### Data Collector
This program can be used to extract data images. It runs a loop, and saves images every
X milliseconds (configured from `CONFIG['miliseconds']`). The folder where the images are saved is 
created in `config.CONFIG['training_sets_path']/CONFIG['training_set_name]`. Folders for each 
type of preprocessing are created, in each folder there are directories named as numbers, 
from 1 to N, where N is the number of classes.

In order to specify for which class the images will be extracted. collect images, which class 
This class should be a numerical value, equal to the `idx+1` in the list 
`config.CONFIG['classes']`. 

##### Example 
Take the following classes defined in `CONFIG['classes']`: `['fist', 'peace', 'one_finger']`.
- To collect images for the "peace" class, set `CONFIG['class_to_collect_data']` to `2`.
- Change the training set name to `"my_gestures"` in `CONFIG['training_set_name']`
- Run `DataCollector.py`. From the window that appears, choose whether to calibrate or not - 
    press "c" to calibrate, press anything else to use default values from `/hsv_ranges.csv`. If
    calibration is chosen, follow the instructions to confirm the HSV values.
- Press "s" to start recording images. Watch the count as it says how many images have been
    created so far. Press as many times as you wish to pause/run
- When satisfied with the number of images, stop the program (hard stop by pressing ctrl+c 
    in the terminal)
- Images are ready! Check the project root, it should look like this:

```
./__training_data
|____./my_gestures
|________./orig
|____________./2 -> contains images
|________./skin_monochrome
|____________./2 -> contains images
|________./skin
|____________./2 -> contains images
|________...
```

To run the data collector, run `DataCollector.py`. 
> Be extra careful, as the data collector will overwrite any images in the directory where 
it saves the new images!


#### Model Trainer
The model trainer trains a CNN using the data provided in one of the `training_data` folders.
The script will transform all data entries into NxN monochrome images, where N is specified by
`CONFIG['training_img_size']`. 
The folder to fetch all images is `CONFIG['training_sets_path']`/`CONFIG['training_set_name']`/
`CONFIG['training_set_image_type']`.
The script will take each folder from this directory and will map those to all labels, specified
in `CONFIG['classes']`. It will be matched on basis `classes[folder_name - 1]`. For example,
all images from folder with name `1` will be the class with index `0`.  
Other params such as batch size, num of epochs, size of test/validation set and so on are also 
set in the CONFIG.

The training script will save the result in `./__results/{timestamp-of-training-start}`. 
It will save:
- The test set, as it is fed into the network (after all preprocessing)
- The actual model and weights
- Model summary, training information (epochs, batch_size etc, test/validation split size etc)
- Confusion matrix for the test data
- Loss/Accuracy graphs 
- Image preprocessing info (size, processing type, etc)

Training time on a MacBook Pro 2017 13" - around 20-30 min.

To run the model trainer, run `ModelTrainer.py`.

#### Predictor
For the predictor to work, the name of the `__results` folder containing the model should be 
provided in the `CONFIG['predictor_model_dir']`. When the predictor runs, it prompts for
calibration, and after that it runs a loop which processes an image each and every frame and
processes the image the same way it has been fed to the model. It takes this information from
the folder with the model automatically, no further adjustments need to be done.

To run the predictor, run `Predictor.py`.


## HPC
I have tried using HPC to train, but it seems I don't ned it. For future reference, to run
on hpc, modify the `deployhpc.sh` file to match the correct ssh destination,
and modify the `exclude-list.txt` to exclude all folders that are not needed for the
trainer to train.

This may not be needed, training times are small due to the images being
only one channel, as well as their size and number (below 100x100px, less than 1000 entrties 
per class).