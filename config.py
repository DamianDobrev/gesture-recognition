import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Prediction parameters.
    # ...
    'predictor_model_dir': '2019-03-27__22-47-28_num_2k_100x100',
    # 'predictor_model_dir': '2019-04-04__16-59-10_first_augm',
    # 'predictor_model_dir': '2019-04-04__19-23-58_second_augm',
    # 'predictor_model_dir': '2019-04-05__03-36-04_augm_100k_imgs',
    # 'predictor_model_dir': '2019-04-07__11-25-26__used_as_run_05',
    # 'predictor_model_dir': '2019-04-07__17-18-12',
    # The certainty percentage for a gestures in order for it
    # to be considered as successfully predicted.
    'predicted_val_threshold': 51.00,

    # Collecting data parameters:
    # ...
    'data_collect_class': 10,
    'data_collect_set_name': 'test2',  # !!! Contents here will be overwritten by the DataCollector.

    # Training parameters.
    # ...
    'num_training_samples': 1000,
    'training_set_name': 'min_1200_per_class',
    'training_set_image_type': 'skin_monochrome',
    'num_epochs': 8,
    'batch_size': 400,
    'training_img_size': 50,
    'test_split': 0.5,
    'validation_split': 0.3,
    'augmentation_on': True,
    'augmentation_count': 2,

    # General. Do not change those.
    # ...
    'training_data_path': os.path.join(package_directory, '__training_data'),
    'results_path': os.path.join(package_directory, '__results'),
    'hsv_ranges_path': os.path.join(package_directory, 'hsv_ranges.csv'),
    'classes': ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock'],
    'size': 200,
    'vis_size': 150,
    'bbox_threshold': 20
}