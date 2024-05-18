from model import resnet18, resnet34, resnet50, resnet101, resnet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, AveragePooling2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle
import os
import time
from datetime import datetime
import wandb 
from wandb.keras import WandbMetricsLogger
import json
from tensorflow import keras

current_time = time.strftime('%Y-%m-%d-%H:%M:%S') #update

parser = ArgumentParser() #update
path_current = os.path.abspath(globals().get("__file__",".")) #update
script_dir = os.path.dirname(path_current) #update
root_path = os.path.abspath(f"{script_dir}/../../..") #update

experiments_dir = os.path.abspath(f"{script_dir}/../../exps/Resnet/experiment_{current_time}") #update
data_path = root_path + "/input/dog-cat-dataset/cats_and_dogs_filtered" #update

if __name__ == "__main__": 
    # Arguments users used when running command lines
    parser.add_argument('--train-folder', default='/kaggle/working/rafdb/train', type=str, help='Where training data is located')
    parser.add_argument('--valid-folder', default='/kaggle/working/rafdb/test', type=str, help='Where validation data is located')
    parser.add_argument('--model', default='resnet50', type=str, help='Type of model')
    parser.add_argument('--num-classes', default=2, type=int, help='Number of classes')
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--image-size', default=224, type=int, help='Size of input image')
    parser.add_argument('--optimizer', default='adam', type=str, help='Types of optimizers')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=120, type=int, help = 'Number of epochs')
    parser.add_argument('--image-channels', default=3, type=int, help='Number channel of input image')
    parser.add_argument('--class-mode', default='sparse', type=str, help='Class mode to compile')
    parser.add_argument('--model-path', default='best_model.h5.keras', type=str, help='Path to save trained model')
    parser.add_argument('--class-names-path', default='class_names.pkl', type=str, help='Path to save class names')
    parser.add_argument('--exp-dir', default = experiments_dir, type = str, help ='folder contain experiemts')

    #use wandb 
    parser.add_argument('--author-name', default='unknown', type=str, help='name of an author')
    parser.add_argument('--use-wandb', default=0, type=int, help='Use wandb')
    parser.add_argument('--wandb-api-key', default = 'fa74a08b8f57907bfa5a6d21d2b665083ed64764', type=str, help='wantdb api key')
    parser.add_argument('--wandb-project-name', default = 'quanproject', type=str, help='name project to store data in wantdb')

    # parser.add_argument('--model-folder', default='.output/', type=str, help='Folder to save trained model')
    args = parser.parse_args()

    #use wandb
    configs = vars(args)
    if(args.author_name == ""):
        raise Exception("author name ??????")
     # Initialize a W&B run
    if args.use_wandb == 1:
        if (args.wandb_api_key ==""):
            raise Exception("if you use Wandb, please entering wandb api key first!")
        if (args.wandb_project_name ==""):
            raise Exception("if you use Wandb, please entering wandb name project first!")
        wandb.login(key=args.wandb_api_key)
        run = wandb.init(
            project = args.wandb_project_name,
            config = configs
        )

    # Project Description

    print('---------------------Welcome to resnet-------------------')
    print('Github: hoangduc199891')
    print('Email: hoangduc199892@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training resnet model with hyper-params:')
    print('===========================')

    # Invoke folder path
    TRAINING_DIR = args.train_folder
    TEST_DIR = args.valid_folder
    
    loss = SparseCategoricalCrossentropy()
    class_mode = args.class_mode
    classes = args.num_classes
        
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(224, 224), batch_size = args.batch_size, class_mode = class_mode )
    val_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size = args.batch_size, class_mode = class_mode)

    class_names=list(train_generator.class_indices.keys())
    with open(args.class_names_path,'wb') as fp:
      pickle.dump(class_names, fp)

    # Create model
    if args.model == 'resnet18':
        model = resnet18(num_classes = classes)
    elif args.model == 'resnet34':
        model = resnet34(num_classes = classes)
    elif args.model == 'resnet50':
        model = resnet50(num_classes = classes)
    elif args.model == 'resnet101':
        model = resnet101(num_classes = classes)
    elif args.model == 'resnet152':
        model = resnet152(num_classes = classes)
    else:
        print('Wrong resnet name, please choose one of these model: resnet18, resnet34, resnet50, resnet101, resnet152')
    # save all arguments to json file
    print(args.exp_dir)
    os.makedirs(args.exp_dir, exist_ok=True)

    args_dict = vars(args)

    # Write dictionary to file
    file_path = os.path.join(experiments_dir, "arguments.json")
    print(file_path)
    with open(file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)
        
    model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))

    if (args.optimizer == 'adam'):
        optimizer = Adam(learning_rate=args.lr)
    elif (args.optimizer == 'sgd'):
        optimizer = SGD(learning_rate=args.lr)
    elif (args.optimizer == 'rmsprop'):
        optimizer = RMSprop(learning_rate=args.lr)
    elif (args.optimizer == 'adadelta'):
        optimizer = Adadelta(learning_rate=args.lr)
    elif (args.optimizer == 'adamax'):
        optimizer = Adamax(learning_rate=args.lr)
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax'


    model.compile(optimizer=optimizer, 
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    # callbacks
    callbacks = []
    save_bestmodel_path = f'{args.exp_dir}/{args.model_path}'
    best_model = ModelCheckpoint(save_bestmodel_path,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    callbacks.append(best_model)
    

    # wandb
    if args.use_wandb == 1:
        cb_wandb = WandbMetricsLogger(log_freq=1)
        callbacks.append(cb_wandb)

    exp_dir = args.exp_dir
    log_file_path = os.path.join(experiments_dir, 'log.csv')

    # Create the experiment directory if it doesn't exist
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Create the log file path
    log_file_path = os.path.join(exp_dir, 'log.csv')
    # logger
    cb_log = CSVLogger(log_file_path)
    callbacks.append(cb_log)

    #Add earlystopping
    callbacks.append(keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=30,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
))
    # Traning
    model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=1,
        validation_data=val_generator,
        callbacks=callbacks)
     #Close your wandb run 
    if args.use_wandb == 1:
        wandb.finish()
    
    pass
