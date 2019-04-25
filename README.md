# Recognizing Hand Gestures 

I have tested this on macOS and it works. I am mostly confident but not completely sure it will work on other OS as well.

## Download prerequisites
1. Download model and metadata needed for the Predictor (~285mb) (please copy paste the link in browser url):
<https://emckclac-my.sharepoint.com/:u:/g/personal/sttb1877_kcl_ac_uk/EWV1BlYtZbxLs4jzZBnXBw8BgcQwK8XwqPSsHRtYJawQGg?e=ok1AOD>
2. Download training data I have used to obtain the results for the report (~2.85GB) (please copy paste the link in browser url):
<https://emckclac-my.sharepoint.com/:u:/g/personal/sttb1877_kcl_ac_uk/EdmJzjufFCZHm-74aDqA74oB4aW_n2wYEAF1F7CkGXQ-YQ?e=TYGCdl>
3. Now, all zips have to be extracted in the SAME directory where the rest of the source code is.
- extract `__results.zip`. This should add: `./__results/{result_name}/...`
- extract `__training_data.zip`. This should add `./__training_data/{dataset_name}/...`
- the source code (and this README.md file) should be in `./`

## Installation and environment setup
0. Open terminal and navigate to the folder where this project is.
1. Make sure python3 is installed `python3 --version`. 
If it is not installed, please install it from here: https://www.python.org/downloads/
2. (if MacOS - skip this step!) `python3 -m pip install --user virtualenv`
3. Depending on the OS:
    - MacOS: `python3 -m venv env`
    - Not MacOS: `python3 -m virtualenv env`
4. `source env/bin/activate` Use this to activate the environment.
5. `pip install -r requirements.txt`

Done. This installs all required packages.

## Run Predictor.
This is the actual hand recognition. Run all these commands inside the virtualenv.

To run with the pretrained model I have created and used, run:
```
python Predictor.py
```

- Note that you will most probably have to calibrate the skin color extraction the first time you run. 
- Please follow the descriptions in the user interface to do that. Make sure all skin pixels are included!
- Make sure to press "s" once you are happy with the calibration, so that the next time the program is ran this calibration is loaded by default.

To run with your own model that you have trained yourself, run:
```
python Predictor.py -m <your_model_name>
```
Where `<your_model_name>` is the name of the folder inside `__results`.

## Run Data Collector.
This, when ran and then started by following the description in the user interface, will collect new training data. 3 notes:
- I have set the config value for the dataset name to be `new_dataset`. This means that all data you create will be saved under `./__training_data/new_dataset`. To change this name, specify it via parameter `-n <your_dataset_name>`.
- I have set the config value for the class for which the program will collect data to be `0` (the 0th class in the classes list: `['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock']`). To change the class, specify it via parameter `-c <your_class_index>`.
- This will overwrite any previous files!
```
python DataCollector.py -n <your_dataset_name> -c <your_class_index>  
```

## Run Model Trainer.
This will train a new model and will create a new unique folder for it inside the `./__results` directory, so don't worry nothing will get overwritten.
```
python ModelTrainer.py -d <your_dataset_name> -n <num_training_samples> -b <batch_size> -a <augmentation_count> -e <num_epochs>
```
All params are optional, without specifying anything it will run with the defaults, which are in `config.py` in the root dir. Alternatively to specifying params, you may find it easier to just modify the values in that file. 

- P.S. if you don't want the data to be augmented, set the param to 0.
- In the end of the training, plots and then confusion matrix will be visualised, please close them when they appear.
- With the default params, it should produce a good result! 

## Running with RealDroneSimulator!
The predictor in conjunction with RealDroneSimulator has only been tested on macOS by me.
1. Download RealDroneSimulator <https://www.realdronesimulator.com/downloads>. On Windows apparently they have an alpha version, for macOS please download the PreAlpha.
2. Run it, choose the lowest resolution (for performance and ease of use) and start it, then when it starts, just press Enter multiple times in order to start the game.
3. The next command will run the predictor in simulator mode, which means it will start sending KEYBOARD events depending on the gestures! Please refer to the requirements section of the report for guides on what the gestures do. Also, it is advised to keep either the prediction window or the RealDroneSimulator focused, because otherwise it may start typing, as it sends keyboard events. To start it, run:
```
python Predictor.py -s
```

Happy predicting! :)

Damyan Dobrev.