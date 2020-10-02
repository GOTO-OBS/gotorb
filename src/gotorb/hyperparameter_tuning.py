import glob
import json
import os

import kerastuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from sklearn.model_selection import train_test_split

from .classifier import get_data_labels, get_data_stamps, STAMPS_FILENAME, LABELS_FILENAME
from .models import bayesian_vgg6_hyperparameterised
from .utils import get_timestamp_str

RANDOM_STATE = 42

def hyperparam_build(hp):
    """
    Take a `keras-tuner` hyperparameters instance and build a model.
    Due to the call signature of the hyperparameter tuning, the limits and choices associated with the hyperparameters
    (hyper-hyper-parameters?) are hard-coded for now.

    :param hp: Hyperparameter optimiser instance.
    :return: Compiled keras-tuner model
    """

    # To get around lack of specifiable constraints within `keras-tuner`, reparameterise model in terms of
    # `block1_size` (number of filters in first conv-conv-pool block) and `scaleup_size`, which when added to
    # `block1_size` gives the number of filters in `block2`. This ensures `block2` >= `block1`, which has big benefits
    # for performance.
    block1_size = hp.Int(name='block1_size', min_value=8, max_value=32, step=8, default=16)
    scaleup_size = hp.Int(name='scale_up', min_value=0, max_value=32, step=8, default=16)
    block2_size = block1_size + scaleup_size

    fc_range = hp.Int(name='fc_size', min_value=64, max_value=512, step=8, default=256)
    init_strat = hp.Choice(name='kernel_init', values=["he_uniform", "glorot_uniform"], default='glorot_uniform')
    dropout_prob = hp.Float(name='dropout_prob', min_value=1e-2, max_value=0.5, sampling='log', default=0.01)

    regulariser_strength = hp.Float(name='regulariser_strength', min_value=1e-8, max_value=1e-1,
                                    sampling='log', default=1e-4)

    activation = hp.Choice(name='activation', values=["ReLU",
                                                      "LeakyReLU",
                                                      "ELU"], default="ReLU")

    model = bayesian_vgg6_hyperparameterised((55, 55, 4), dropout_prob=dropout_prob, block1_size=block1_size,
                                             block2_size=block2_size, fc_size=fc_range, initialiser=init_strat,
                                             kern_size=3, regulariser_strength=regulariser_strength,
                                             activation=activation)

    lr_range = hp.Float(name='learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    opt = tf.keras.optimizers.Adam(lr=lr_range)

    # include the AUC here as a metric, so that the tuning code can use it as an objective.
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


def run_model_tuning(data_dir, max_epochs=500, patience=10, batch_size=64, validation_split=0.1, test_split=0.1,
                     layers=(0,1,2,3), max_model_pars=1000000, runname=None, hyperband_iters=2, metric='val_accuracy'):
    """
    Wrapper for keras-tuner using Hyperband optimisation to find optimal hyperparameters. Uses a stripped-down version
    of `classifier.train` to mirror the load-in and preprocessing as best as possible.

    Bayesian optimisation uses a fixed number of trials to find the best solution in a given resource budget.

    :param data_dir: Location of training data files
    :param max_epochs: Maximum number of epochs to train for
    :param patience: Number of epochs without improvement in validation loss before terminating.
    :param batch_size: Batch size to evaluate with
    :param validation_split: Proportion of data to allocate to validation
    :param test_split: Proportion of data to allocate to test - these purely for consistency w/ main training
    :param layers: layers to use
    :param max_model_pars: maximum model complexity (trainable params)
    :param runname: User-selected name for a given tuning run. Defaults to `None`, where a timestamp will be used
    :param hyperband_iters: Number of full iterations of the Hyperband algorithm to perform
    :return: None
    """
    # try to catch annoying tensorflow errors.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # catch warnings for cleaner output

    # enable resuming previous runs for fault tolerance
    if runname is None:
        timestamp = get_timestamp_str()
        runname = "hprun_" + timestamp

    working_dir_loc = data_dir

    # load in training and validation data
    print('loading in training data')
    data_stamps = get_data_stamps(os.path.join(data_dir, STAMPS_FILENAME))
    data_labels = get_data_labels(os.path.join(data_dir, LABELS_FILENAME))

    data_stamps = data_stamps[:, :, :, layers]

    print('normalising stamps')
    i = 0
    while True:
        j = min(i + 25000, data_stamps.shape[0])
        data_stamps[i:j, :, :, :] = np.array(tf.keras.backend.l2_normalize(data_stamps[i:j, :, :, :], axis=(1, 2)))
        i = j
        if i == data_stamps.shape[0]:
            break

    x_train, x_test, y_train, y_test = train_test_split(data_stamps,
                                                        data_labels.label,
                                                        test_size=test_split,
                                                        random_state=RANDOM_STATE,
                                                        )

    num_training_examples_per_class = np.array([len(y_train) - np.sum(y_train), np.sum(y_train)])
    weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
    normalised_weight = weights / np.max(weights)

    class_weight = {i: w for i, w in enumerate(normalised_weight)}

    # init hyperparameter tuner
    objective = kt.Objective(metric, direction="max") # custom objective - maximise chosen metric

    tuner = kt.Hyperband(hyperparam_build, objective=objective, max_epochs=max_epochs,
                                    directory=working_dir_loc, project_name=runname, seed=RANDOM_STATE,
                                    max_model_size=max_model_pars, factor=np.e, hyperband_iterations=hyperband_iters)

    data_augmentation = {'horizontal_flip': True, 'vertical_flip': True} # hard-code this
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_augmentation,
                                                                  validation_split=validation_split)
    training_generator = data_generator.flow(x_train, y_train, batch_size=batch_size,
                                             subset='training', seed=RANDOM_STATE)
    validation_generator = data_generator.flow(x_train, y_train, batch_size=batch_size,
                                               subset='validation', seed=RANDOM_STATE)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True,)

    tuner.search(training_generator,
                          steps_per_epoch=len(x_train) * (1 - validation_split) // batch_size,
                          validation_data=validation_generator,
                          validation_steps=(len(x_train) * validation_split) // batch_size,
                          class_weight=class_weight,
                          epochs=max_epochs,
                          callbacks=[early_stopping],
                          verbose=1)

    best_hps = tuner.get_best_hyperparameters()[0]

    for par in best_hps.values:
        print("{}: {}".format(par, best_hps.values[par]))

    return best_hps


def parse_kerastuner_file(workdir):
    """
    Parse all trials from a given `keras-tuner` run into a Pandas dataframe, for evaluation and plotting.
    Specifically for use with the functions in `.hyperparameter_tuning`, but can be adapted to parse general files also.

    :param workdir: Location containing the trial folders
    :return: Pandas dataframe containing all trial results.
    """

    full_meta = [] # init empty container to gather Pandas dataframe slices

    print("Parsing model files")
    for tid in tqdm.tqdm(glob.glob(workdir + "/trial_*")):
        jsonfile = json.load(open(tid + "/trial.json", "rb"))
        if jsonfile["score"] is not None:
            jsonfile["hyperparameters"]["values"]["tuner/trial_id"] = jsonfile["trial_id"]
            meta = jsonfile["hyperparameters"]["values"]

            # build a model to get param size.
            trial_model = bayesian_vgg6_hyperparameterised((55, 55, 4),
                                                           dropout_prob=meta["dropout_prob"],
                                                           block1_size=meta["block1_size"],
                                                           block2_size=meta["block1_size"] + meta["scale_up"],
                                                           fc_size=meta["fc_size"],
                                                           initialiser=meta["kernel_init"])

            totalpars = tf.reduce_sum([tf.reduce_prod(r.shape) for r in trial_model.get_weights()]).numpy()
            del trial_model

            meta["params"] = totalpars
            meta["score"] = jsonfile["score"]
            meta["block2_size"] = meta["block1_size"] + meta["scale_up"]
            meta.pop("scale_up")
            full_meta.append(meta)

    all_trials = pd.DataFrame(full_meta)

    return all_trials