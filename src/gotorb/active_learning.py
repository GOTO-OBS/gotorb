# Active learning tools for optimal training set construction

import os

import numpy as np
import pandas as pd
import tables
import tensorflow as tf

from .classifier import dropout_pred, get_data_stamps, get_data_labels


def binary_entropy(p, eps=1e-6):
    tempval = np.copy(p)  # don't overwrite array
    tempval[tempval < eps] = eps
    tempval[tempval > (1 - eps)] = 1 - eps

    return -tempval * np.log2(tempval) - (1 - tempval) * np.log2(1 - tempval)

def bald_strategy_step(model_dir, pool_dir, out_dir, n_acquire, n_samples=30):
    """
    Routine for Bayesian Active Learning by Disagreement, a (TBC) optimal strategy for building a training set
    REF: Houlsby et al., (2011) - https://arxiv.org/abs/1112.5745
    REF: Gal et al., (2017) - https://arxiv.org/abs/1703.02910

    Running this code evaluates a given model on a pool of labelled stamps, and appends the N top scoring examples to
    the existing dataset, in a new output directory. Also written is a bool mask so that future acquisition steps cannot
    select the same points twice. This keeps the full pool available for other acquisition strategies.

    :param model_dir: location of the chosen model
    :param pool_dir: location of a bulk set of stamp files to act as pool
    :param out_dir: where to write the outputs to, relative to the base directory (MODELDIR/../../)
    :param n_acquire: number of stamps to acquire from the pool
    :param n_samples: number of posterior samples to use for inference.
    :return: None
    """

    modeldir = model_dir
    basedir = modeldir + "../../"
    poolpath = pool_dir

    newdir_bald = os.path.join(basedir, out_dir)

    if os.path.exists(newdir_bald):
        print("Path already exists!")
    else:
        os.makedirs(newdir_bald)
        print("Making directories")

    des_acq_num = n_acquire

    print("Loading pool")
    model = tf.keras.models.load_model(modeldir + "model.h5")
    pool_stamps = get_data_stamps(os.path.join(poolpath, "data_stamps.h5"))
    pool_labels = get_data_labels(os.path.join(poolpath, "data_labels.csv"))

    acq_num = min(des_acq_num, len(pool_stamps))  # if we don't have enough stamps, protect from duplication

    # load in mask files here if they exist:

    try:
        bald_mask = np.squeeze(pd.read_csv(os.path.join(modeldir, "../", "baldmask.csv")).iloc[:, 0].values)
        print("maskfile exists, loaded")
    except FileNotFoundError:
        print("no maskfiles found, generating blanks")
        bald_mask = np.ones(len(pool_labels), dtype=bool)

    print("L2 norming stamps")
    norm_stamps = tf.keras.backend.l2_normalize(pool_stamps[:, :, :, (0, 1, 2, 3)], axis=(1, 2))

    print("Evaluating posteriors")
    posteriors = dropout_pred(model, norm_stamps, verbose=1, nsamples=n_samples)
    bald_score = binary_entropy(np.average(posteriors, axis=1)) - np.average(binary_entropy(posteriors), axis=1)

    # Select maximum BALD values and set the respective mask to False
    valid_idxs_bald = np.squeeze(np.argwhere(bald_mask))
    baldsortidx = np.argsort(-bald_score)
    chosen_idx = [x for x in baldsortidx if x in valid_idxs_bald][:acq_num]  # this is inefficient but works
    bald_mask[chosen_idx] = False

    testdf = pd.DataFrame(bald_mask)
    testdf.to_csv(os.path.join(newdir_bald, "baldmask.csv"))

    print("Loading existing datastamps")
    data_stamps = get_data_stamps(os.path.join(modeldir, "../", "data_stamps.h5"))
    data_labels = get_data_labels(os.path.join(modeldir, "../", "data_labels.csv"))
    supp_stamps = pool_stamps[chosen_idx]
    new_stamps = np.concatenate([data_stamps, supp_stamps])

    supp_labels = pool_labels.iloc[chosen_idx]
    new_labels = pd.concat((data_labels, supp_labels))

    print("Writing stamps and labels to {}".format(out_dir))
    new_labels.to_csv(os.path.join(newdir_bald, "data_labels.csv"))

    h5_file = os.path.join(newdir_bald, "data_stamps.h5")
    h5_out = tables.open_file(h5_file, mode='a')
    atom = tables.Float64Atom()
    if '/src' not in h5_out:
        array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, *np.shape(new_stamps[0, :, :, :])))
    else:
        array_c = h5_out.root.src

    array_c.append(new_stamps)
    h5_out.close()
    print("Done!")
    return None

def random_acquisition_strategy(model_dir, pool_dir, out_dir, n_acquire):
    """
    Naive random acquisition strategy - selects a random N stamps from a pool of stamps and appends them, creating a new
    dataset at out_dir

    :param model_dir: location of the chosen model
    :param pool_dir: location of a bulk set of stamp files to act as pool
    :param out_dir: where to write the outputs to, relative to the base directory (modeldir/../../)
    :param n_acquire: number of stamps to acquire from the pool
    :return: None
    """

    modeldir = model_dir
    basedir = modeldir + "../../"
    poolpath = pool_dir

    newdir_rand = os.path.join(basedir, out_dir)

    if os.path.exists(newdir_rand):
        print("Path already exists!")
    else:
        os.makedirs(newdir_rand)
        print("Making directories!")

    des_acq_num = n_acquire # desired n_acquire might be different to actual if we run out of stamps

    # Ingest data
    print("Loading pool")
    pool_stamps = get_data_stamps(os.path.join(poolpath, "data_stamps.h5"))
    pool_labels = get_data_labels(os.path.join(poolpath, "data_labels.csv"))

    acq_num = min(des_acq_num, len(pool_stamps))  # if we don't have enough stamps, protect from duplication

    # load in mask files here if they exist:
    try:
        rand_mask = np.squeeze(pd.read_csv(os.path.join(modeldir, "../", "randmask.csv")).iloc[:, 0].values)
        print("maskfile exists, loaded")
    except FileNotFoundError:
        print("no maskfiles found, generating blanks")
        rand_mask = np.ones(len(pool_labels), dtype=bool)

    # Select random values
    valid_idxs_RAND = np.squeeze(np.argwhere(rand_mask))
    idxs_orig = np.arange(0, len(pool_labels))
    idxs_shuf = np.random.permutation(idxs_orig)

    chosen_idx = [x for x in idxs_shuf if x in valid_idxs_RAND][:acq_num]
    rand_mask[chosen_idx] = False

    testdf = pd.DataFrame(rand_mask)
    testdf.to_csv(os.path.join(newdir_rand, "randmask.csv"))

    data_stamps = get_data_stamps(os.path.join(modeldir, "../", "data_stamps.h5"))
    data_labels = get_data_labels(os.path.join(modeldir, "../", "data_labels.csv"))

    supp_stamps = pool_stamps[chosen_idx]
    new_stamps = np.concatenate([data_stamps, supp_stamps])

    supp_labels = pool_labels.iloc[chosen_idx]
    new_labels = pd.concat((data_labels, supp_labels))

    print("Writing stamps and labels to {}".format(out_dir))
    new_labels.to_csv(os.path.join(newdir_rand, "data_labels.csv"))

    h5_file = os.path.join(newdir_rand, "data_stamps.h5")
    h5_out = tables.open_file(h5_file, mode='a')
    atom = tables.Float64Atom()
    if '/src' not in h5_out:
        array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, *np.shape(new_stamps[0, :, :, :])))
    else:
        array_c = h5_out.root.src

    array_c.append(new_stamps)
    h5_out.close()

    print("Done!")
    return None