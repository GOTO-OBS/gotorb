import logging
import os
from itertools import product

import hickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from astropy.visualization import ZScaleInterval
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import backend as workaround_backend
from tqdm import tqdm
from tqdm.contrib.itertools import product as tqdmproduct

from .label_data import LABELS_FILENAME, STAMPS_FILENAME
from .models import vgg6
from .utils import get_timestamp_str

LOGGING_FORMAT = '%(asctime)s  %(levelname)-10s %(processName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

MODELS_FILENAME = 'model.h5'
META_FILENAME = 'meta.hkl'


def setup_logging(out_dir, logging_fn):
    fh = logging.FileHandler(os.path.join(out_dir, logging_fn))
    fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(fh)


def get_data_labels(filename):
    return pd.read_csv(filename)


def get_data_stamps(filename):
    f = tables.open_file(filename, mode='r', driver='H5FD_CORE')
    data_stamps = np.array(f.root.src)
    f.close()
    return data_stamps


def train(data_dir, model=None, layers=None, optimizer=None, normalise='l2', test_split=0.2, validation_split=0.2,
          class_weight=True, loss='binary_crossentropy', batch_size=64, data_augmentation=None,
          epochs=100, monitor='val_loss', patience=10, random_state=42, fit_kwargs=None, tensorboard=True):
    """
    Training utility for classifier, handling data preprocessing, partitioning, augmentation, and training.

    :param data_dir: directory containing output from `label_data.make()`
    :param model: callable accepting `input_shape` as first param and returning a keras model.
                  if none uses `models.vgg6`
    :param layers: list of layer indices to use in model from data stamps (e.g. [0,1,2]). if none uses all.
    :param optimizer: a kera optimizer to be used by the training. if none defaults to `keras.optimizers.Adam` with
                      learning rate of 3e-5.
    :param normalise: callable accepting a stamp layer that returns a normalised varsion of the stamp. default 'l2' will
                      use keras' l2-normalisation.
    :param test_split: fraction of input data to use to test the model
    :param validation_split: fraction of training data to use for validation while training
    :param class_weight: whether to weight class labels by their frequency in the input data
    :param loss: the loss score measured by the keras model.
    :param batch_size: directly passed to keras model, see keras docs
    :param data_augmentation: dictionary of data_augementation to be applied to input data, keys must be valid arguments
                              to `keras.preprocessing.image.ImageDataGenerator`.
    :param epochs: number of epochs to train model for, see keras docs
    :param monitor: the value to monitor for patience, see keras docs
    :param patience: patience value, see keras docs
    :param random_state: random seed for reproducibility
    :param fit_kwargs: dict of extra kwargs to pass to `keras.model.fit_generator' or `keras.model.fit`
    :param tensorboard: if true, writes `tensorboard` log files to `$data_dir/tflog` to attach a tensorboard instance to.
    :return: model (trained tensorflow Model instance), meta (dictionary containing metadata and train/test data)
    """
    labels_filename = os.path.join(data_dir, LABELS_FILENAME)
    stamps_filename = os.path.join(data_dir, STAMPS_FILENAME)
    for fn in [labels_filename, stamps_filename]:
        try:
            open(fn)
        except FileNotFoundError:
            raise ValueError('Could not find {}'.format(fn))

    timestamp = get_timestamp_str()
    out_dir = os.path.join(data_dir, 'model_{}'.format(timestamp))
    os.makedirs(out_dir)
    setup_logging(out_dir, 'model_{}.log'.format(timestamp))
    logger.info('output to be stored in {}'.format(out_dir))

    logger.info('reading data labels from {}'.format(labels_filename))
    data_labels = get_data_labels(labels_filename)
    logger.info('reading data stamps from {}'.format(stamps_filename))
    data_stamps = get_data_stamps(stamps_filename)

    # log data proportions
    logger.info("Training set breakdown:")
    if "metalabel" in data_labels:
        metalabels = data_labels["metalabel"]

        for m in np.unique(metalabels):
            logger.info("{}: {} of {}".format(m, np.sum(metalabels == m), len(metalabels)))

    else:
        logger.info("Metalabels unavailable as using old version dataset")

    if layers:
        logger.info('using only layers {}'.format(layers))
        data_stamps = data_stamps[:, :, :, layers]
    else:
        logger.info('using all layers')

    if normalise == 'l2':
        logger.info('L2 normalising src stamps')
        i = 0
        while True:
            j = min(i + 25000, data_stamps.shape[0])
            data_stamps[i:j, :, :, :] = np.array(keras.backend.l2_normalize(data_stamps[i:j, :, :, :], axis=(1, 2)))
            i = j
            logger.info('done {}/{}'.format(i, data_stamps.shape[0]))
            if i == data_stamps.shape[0]:
                break
    elif callable(normalise):
        logger.info('custom normalising src stamps')
        data_stamps = normalise(data_stamps)

    x_train, x_test, y_train, y_test = train_test_split(data_stamps,
                                                        data_labels.label,
                                                        test_size=test_split,
                                                        random_state=random_state,
                                                        )
    _, _, mask_train, mask_test = train_test_split(data_labels.label,
                                                   list(range(len(data_labels.label))),
                                                   test_size=test_split,
                                                   random_state=random_state,
                                                   )
    masks = {'training': mask_train, 'test': mask_test}
    logger.info('n_train = {}, n_test = {}'.format(len(x_train), len(x_test)))

    keras.backend.clear_session()
    early_stopping = keras.callbacks.EarlyStopping(monitor=monitor,
                                                   patience=patience,
                                                   restore_best_weights=True,
                                                   )

    binary_classification = True if loss == 'binary_crossentropy' else False
    n_classes = 1 if binary_classification else 2
    logger.info('n_classes = {}'.format(n_classes))

    if class_weight:
        if binary_classification:
            num_training_examples_per_class = np.array([len(y_train) - np.sum(y_train), np.sum(y_train)])
        else:
            num_training_examples_per_class = np.sum(y_train, axis=0)
        assert 0 not in num_training_examples_per_class, 'found class without any examples!'

        weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
        normalised_weight = weights / np.max(weights)

        class_weight = {i: w for i, w in enumerate(normalised_weight)}
        logger.info('class weights = {}'.format(class_weight))
    else:
        class_weight = None

    input_shape = x_train.shape[1:]
    logger.info('input shape = {}'.format(input_shape))

    if model is None:
        model = vgg6
        logger.info('using default vgg6 model')
    model = model(input_shape, n_classes=n_classes)

    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr=3e-5)
        logger.info('using default adam optimizer')

    callbacks = [early_stopping]

    if tensorboard:
        tflog_dir = os.path.join(out_dir, "tflog/")
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tflog_dir, profile_batch=0, histogram_freq=1))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    strlist = []
    model.summary(print_fn=strlist.append)
    logger.info('\n'.join(strlist))

    if fit_kwargs is None:
        fit_kwargs = dict()

    try:
        if data_augmentation is not None:
            logger.info('using data augmentation: {}'.format(data_augmentation))
            data_generator = keras.preprocessing.image.ImageDataGenerator(**data_augmentation,
                                                                          validation_split=validation_split, )
            training_generator = data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training')
            validation_generator = data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation')
            logger.info('fitting with `fit_generator`')
            h = model.fit(training_generator,
                          steps_per_epoch=len(x_train) * (1 - validation_split) // batch_size,
                          validation_data=validation_generator,
                          validation_steps=(len(x_train) * validation_split) // batch_size,
                          class_weight=class_weight,
                          epochs=epochs,
                          verbose=1,
                          callbacks=callbacks,
                          **fit_kwargs,
                          )
            history = h.history
        else:
            logger.info('fitting with `fit`')
            h = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          validation_split=validation_split,
                          class_weight=class_weight,
                          epochs=epochs,
                          verbose=1,
                          callbacks=callbacks,
                          **fit_kwargs,
                          )
            history = h.history

    except KeyboardInterrupt:
        logger.info('stopping due to keyboard interrupt')
        history = model.history.history

    model_h5_filename = os.path.join(out_dir, MODELS_FILENAME)
    model.save(model_h5_filename)
    logger.info('trained model saved as {}'.format(model_h5_filename))

    logger.info('saving model meta data')
    meta = dict(
        timestamp=timestamp,
        batch_size=batch_size,
        epochs=len(history["val_accuracy"]),
        patience=patience,
        loss=loss,
        test_split=test_split,
        validation_split=validation_split,
        class_weight=class_weight,
        data_augmentation=data_augmentation,
        fit_kwargs=fit_kwargs,
        monitor=monitor,
        random_state=random_state,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        masks=masks,
        layers=layers,
        labels_filename=os.path.abspath(labels_filename),
        stamps_filename=os.path.abspath(stamps_filename),
        model_dir=os.path.abspath(out_dir),
        history=history,
    )
    model_meta_filename = os.path.join(out_dir, META_FILENAME)
    hickle.dump(meta, model_meta_filename, mode='w')
    logger.info('model meta data saved as {}'.format(model_meta_filename))

    return model, meta


def evaluate(model, meta, plot_fmt='png', plot_examples=False):
    """
    Evaluate the performance of a model with some printed statistics and plots.

    Can be called using the `model, meta` output from `classifier.train()` or by pointing
    it to their filepaths.

    :param plot_examples:
    :param model: filepath to a trained keras model file, or a kera model itself
    :param meta: filepath to a trained models meta pickle file, or the meta data itself
    :param plot_fmt: format for plots, must be understandable by matplotlib
    :return: None
    """

    if isinstance(model, str):
        logger.info('loading model')
        model = keras.models.load_model(model)
    if isinstance(meta, str):
        logger.info('loading meta')
        meta = hickle.load(meta)

    data_labels = get_data_labels(meta['labels_filename'])

    model_dir = meta['model_dir']
    setup_logging(model_dir, 'model_{}.log'.format(meta['timestamp']))

    for val in ['batch_size', 'epochs', 'patience', 'loss', 'test_split', 'validation_split',
                'class_weight', 'data_augmentation', 'fit_kwargs', 'monitor', 'random_state',
                'layers', 'labels_filename', 'stamps_filename']:
        logger.info('{}: {}'.format(val, meta[val]))
    logger.info('optimizer type: {}'.format(model.optimizer))

    logger.info('evaluating training set performance')
    train_pred = model.predict(meta['x_train'], batch_size=meta['batch_size'], verbose=1)
    misclassified_train_mask = (np.array(list(map(int, data_labels.label[meta['masks']['training']]))).flatten() ^
                                np.array(list(map(int, np.rint(train_pred)))).flatten())
    misclassified_train_mask = [ii for ii, mi in enumerate(misclassified_train_mask) if mi == 1]
    train_numbers = data_labels.index.values[meta['masks']['training']]
    train_labels = data_labels.label.values[meta['masks']['training']]
    misclassifications_train = {int(c): [int(l), float(p)]
                                for c, l, p in zip(train_numbers[misclassified_train_mask],
                                                   train_labels[misclassified_train_mask],
                                                   train_pred[misclassified_train_mask])}

    logger.info('evaluating test set performance')
    test_pred = model.predict(meta['x_test'], batch_size=meta['batch_size'], verbose=1)
    misclassified_test_mask = (np.array(list(map(int, data_labels.label[meta['masks']['test']]))).flatten() ^
                               np.array(list(map(int, np.rint(test_pred)))).flatten())
    misclassified_test_mask = [ii for ii, mi in enumerate(misclassified_test_mask) if mi == 1]
    test_numbers = data_labels.index.values[meta['masks']['test']]
    test_labels = data_labels.label.values[meta['masks']['test']]
    misclassifications_test = {int(c): [int(l), float(p)]
                               for c, l, p in zip(test_numbers[misclassified_test_mask],
                                                  test_labels[misclassified_test_mask],
                                                  test_pred[misclassified_test_mask])}
    allclassifications_test = {int(c): [int(l), float(p)]
                               for c, l, p in zip(test_numbers,
                                                  test_labels,
                                                  test_pred)}
    test_class = np.rint(test_pred)
    test_eval = model.evaluate(meta['x_test'], meta['y_test'], batch_size=meta['batch_size'], verbose=1)
    test_loss = float(test_eval[0])
    test_accuracy = float(test_eval[1])

    logger.info('creating confusion matrices')
    cm = confusion_matrix(meta['y_test'], test_class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    preds = test_pred
    labels = data_labels.label[meta['masks']['test']]

    logger.info('making plots')
    _plot_validation(meta['history'], os.path.join(model_dir, 'plot_validation.{}'.format(plot_fmt)))
    _plot_confusion_matrix(cm, os.path.join(model_dir, 'plot_confusionmatrix.{}'.format(plot_fmt)))
    _plot_fpr(preds, labels, os.path.join(model_dir, 'plot_fpr.{}'.format(plot_fmt)))
    _plot_roc(preds, labels, os.path.join(model_dir, 'plot_roc.{}'.format(plot_fmt)))
    _plot_probdistn(preds, labels, os.path.join(model_dir, 'plot_probdistn.{}'.format(plot_fmt)))

    if plot_examples:
        data_stamps = get_data_stamps(meta['stamps_filename'])
        _plot_examples(misclassifications_test, data_stamps,
                       os.path.join(model_dir, 'plot_misclassifiedexamples.{}'.format(plot_fmt)))
        _plot_examples(allclassifications_test, data_stamps,
                       os.path.join(model_dir, 'plot_randomexamples.{}'.format(plot_fmt)))

    logger.info('confusion matrix')
    logger.info(cm)
    logger.info('confusion matrix normalised')
    logger.info(cm_norm)
    logger.info('test lost = {:.4f}, test accuracy = {:.4f}'.format(test_loss, test_accuracy))
    logger.info('#train misclassifications = {}'.format(len(misclassifications_train)))
    logger.info('#test misclassifications = {}'.format(len(misclassifications_test)))


def benchmark(model, benchdir):
    """
    Test a trained model on a benchmark set.

    :param model: folder containing trained keras model.h5
    :param benchdir: directory containing data labels and datastamps - output like add_marshall_cands
    :return: None
    """
    labels_filename = os.path.join(benchdir, LABELS_FILENAME)
    stamps_filename = os.path.join(benchdir, STAMPS_FILENAME)

    if isinstance(model, str):
        logger.info('loading model')
        model = keras.models.load_model(model)

    logger.info('reading data labels from {}'.format(labels_filename))
    data_labels = get_data_labels(labels_filename)
    class_labels = data_labels.label

    logger.info('reading data stamps from {}'.format(stamps_filename))
    data_stamps = get_data_stamps(stamps_filename)

    # use layers from model
    data_stamps = data_stamps[:, :, :, (0, 1, 2)]

    # l2 normalise the stamps for consistence
    i = 0
    while True:
        j = min(i + 25000, data_stamps.shape[0])
        data_stamps[i:j, :, :, :] = np.array(keras.backend.l2_normalize(data_stamps[i:j, :, :, :], axis=(1, 2)))
        i = j
        logger.info('done {}/{}'.format(i, data_stamps.shape[0]))
        if i == data_stamps.shape[0]:
            break

    logger.info('benchmarking on {} framestacks'.format(len(class_labels)))
    bench_score = model.predict(data_stamps, batch_size=64, verbose=1)
    bench_class = np.rint(bench_score)

    cm = confusion_matrix(class_labels, bench_class)

    _plot_fpr(bench_score, class_labels, 'bench_fpr.png')
    _plot_roc(bench_score, class_labels, 'bench_roc.png')
    _plot_probdistn(bench_score, class_labels, 'bench_pdist.png')
    _plot_confusion_matrix(cm, 'bench_confusionmatrix.png')


def dropout_pred(model, stamps, nsamples=20, batchsize=64, verbose=False):
    """
    Returns the posteriors associated with a given set of stamps using Monte Carlo dropout

    :param verbose: Print a progress bar
    :param model: trained keras model instance
    :param stamps: input L2 normalised stamps
    :param nsamples: number of samples drawn from the posterior
    :param batchsize: size of evaluation batches
    :return: 1D array of predictions if input is 3D, else an array Nstamps x Nsamples containing predictions
    """

    # Re-enable dropout at predict time - needs workaround_backend to work with tf>=2.1.0
    eval_model = keras.backend.function(
        [model.layers[0].input,
         workaround_backend.symbolic_learning_phase()],
        [model.output])

    # immediately evaluate single stamps - need to make them 4D.
    if len(np.shape(stamps)) == 3:
        output = np.squeeze([eval_model([stamps[np.newaxis, :], bool(1)]) for _ in range(nsamples)])
        return output

    # init an array to hold our predictions
    nstamps = len(stamps)
    output = np.zeros((nstamps, nsamples))

    # cartesian product used to remove the nested for loop - is it quicker, who knows?
    if verbose:
        for tup in tqdmproduct(np.arange(0, nstamps, batchsize), np.arange(0, nsamples)):
            idx, j = tup
            output[idx:min(idx + batchsize, nstamps), j] = np.squeeze(
                eval_model([stamps[idx:min(idx + batchsize, nstamps), :, :, :], bool(1)]))
    else:
        for tup in product(np.arange(0, nstamps, batchsize), np.arange(0, nsamples)):
            idx, j = tup
            output[idx:min(idx + batchsize, nstamps), j] = np.squeeze(
                eval_model([stamps[idx:min(idx + batchsize, nstamps), :, :, :], bool(1)]))

    return output

def latentvec_pred(model, stamps, batchsize=64, verbose=False, layer='auto'):
    """
    Returns the output of a given hidden layer for visualisation - defaults to idx -4, the flattened feature maps

    :param verbose: Print a progress bar
    :param model: trained keras model instance
    :param stamps: input L2 normalised stamps
    :param batchsize: size of evaluation batches
    :param layer: either "auto", which finds the flattened layer automatically, or any int < model size.
    :return: 2D array, Nstamps x Nfeatures
    """
    if layer == 'auto':
        print("Auto-selecting flattened layer")
        selected_layer = [i for i in range(len(model.layers)) if model.layers[i].name == 'flatten'][0]
    elif (type(layer) == int) & (layer < len(model.layers)):
        print("Using layer {} output".format(layer))
        selected_layer = layer
    else:
        raise ValueError("Invalid layer specified.")

    # Init a Keras function that takes model input and returns the output of a given hidden layer
    eval_model = keras.backend.function([model.layers[0].input], [model.layers[selected_layer].output])

    # If single stamp, we should make 4D tensor and evaluate immediately.
    if len(np.shape(stamps)) == 3:
        output = eval_model(stamps[np.newaxis, 0])[0]
        return output

    # if we are evaluating multiple stamps, init a storage array for outputs N stamps x N hidden layer neurons
    nstamps = len(stamps)
    output = np.zeros((nstamps, model.layers[selected_layer].output.shape[1]))

    if verbose:
        for idx in tqdm(range(0, nstamps, batchsize)):
            batch = stamps[idx:min(idx + batchsize, nstamps), :]
            output[idx:min(idx + batchsize, nstamps), :] = eval_model(batch)[0]
    else:
        for idx in range(0, nstamps, batchsize):
            batch = stamps[idx:min(idx + batchsize, nstamps), :]
            output[idx:min(idx + batchsize, nstamps), :] = eval_model(batch)[0]

    return output

def _plot_validation(history, plot_filename):
    plt.clf()
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(121)
    ax.plot(history['accuracy'], label='Training', linewidth=1.2)
    ax.plot(history['val_accuracy'], label='Validation', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    ax.grid(True, linewidth=.3)

    ax2 = fig.add_subplot(122)
    ax2.plot(history['loss'], label='Training', linewidth=1.2)
    ax2.plot(history['val_loss'], label='Validation', linewidth=1.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')
    ax2.grid(True, linewidth=.3)
    plt.tight_layout()
    plt.savefig(plot_filename)


def _plot_confusion_matrix(cm, plot_filename, classes=('Bogus', 'Real')):
    plt.clf()
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.set(xticks=np.arange(cm_norm.shape[1]),
           yticks=np.arange(cm_norm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title='Confusion matrix',
           ylabel='True class',
           xlabel='Predicted class',
           ylim=(1.5, -0.5),
           )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, '{:.2f}% ({:d})'.format(cm_norm[i, j], cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    size=10)

    plt.tight_layout()
    plt.savefig(plot_filename)


def _plot_fpr(preds, labels, plot_filename):
    rbbins = np.arange(-0.0001, 1.0001, 0.0001)
    h_b, e_b = np.histogram(preds[(labels == 0).values], bins=rbbins, density=True)
    h_b_c = np.cumsum(h_b)
    h_r, e_r = np.histogram(preds[(labels == 1).values], bins=rbbins, density=True)
    h_r_c = np.cumsum(h_r)

    fig, ax = plt.subplots(dpi=300)

    rb_thres = np.array(list(range(len(h_b)))) / len(h_b)

    ax.plot(rb_thres, h_r_c / np.max(h_r_c),
            label='False Negative Rate (FNR)', linewidth=1.5)
    ax.plot(rb_thres, 1 - h_b_c / np.max(h_b_c),
            label='False Positive Rate (FPR)', linewidth=1.5)

    mmce = (h_r_c / np.max(h_r_c) + 1 - h_b_c / np.max(h_b_c)) / 2
    ax.plot(rb_thres, mmce, '--',
            label='Mean misclassification error', color='gray', linewidth=1.5)

    ax.set_xlim([-0.05, 1.05])

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_yscale('log')
    ax.set_ylim([5e-4, 1])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) if x < 0.01 else '{:,.0%}'.format(x) for x in vals])

    # thresholds:
    thrs = [0.5, 0.75]
    for t in thrs:
        m_t = rb_thres < t
        fnr = np.array(h_r_c / np.max(h_r_c))[m_t][-1]
        fpr = np.array(1 - h_b_c / np.max(h_b_c))[m_t][-1]
        ax.vlines(t, 0, max(fnr, fpr))
        ax.text(t - .05, max(fnr, fpr) + 0.01, f' {fnr * 100:.1f}% FNR\n {fpr * 100:.1f}% FPR', fontsize=10)

    ax.set_xlabel('RB score threshold')
    ax.set_ylabel('Cumulative percentage')
    ax.legend(loc='lower center')
    ax.grid(True, which='major', linewidth=.5)
    ax.grid(True, which='minor', linewidth=.3)

    plt.tight_layout()
    plt.savefig(plot_filename)


def _plot_roc(preds, labels, plot_filename):
    fig, (ax, ax2) = plt.subplots(1, 2, dpi=300)
    # fig.subplots_adjust(bottom=0.09, left=0.05, right=0.70, top=0.98, wspace=0.2, hspace=0.2)

    ax.plot([0, 1], [0, 1], color='#333333', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Contamination)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.grid(True, linewidth=.3)

    ax2.set_xlim([0.0, .2])
    ax2.set_ylim([0.8, 1.005])
    ax2.set_xlabel('False Positive Rate (Contamination)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    # ax.legend(loc="lower right")
    ax2.grid(True, linewidth=.3)

    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2)
    ax2.plot(fpr, tpr, lw=2, label=f'ROC (area = {roc_auc:.5f})')
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(plot_filename)


def _plot_probdistn(preds, labels, plot_filename):
    rbbins = np.arange(-0.0, 1.02, 0.02)
    fig, ax = plt.subplots(dpi=300)

    ax.hist(preds[labels == 0], histtype='step', bins=rbbins, label='bogus', lw=2)
    ax.hist(preds[labels == 1], histtype='step', bins=rbbins, label='real', lw=2)

    ax.set_xlim([-0.01, 1.01])

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yscale('log')

    ax.set_xlabel('RB score')
    ax.set_ylabel('N')
    ax.legend(loc='upper center')
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(plot_filename)


def _plot_examples(classifications, data_stamps, plot_filename, n=12):
    # classification dicts are: {data_number: [label, model_score].. }
    real = [(i, sc[0], sc[1]) for i, sc in classifications.items() if sc[0] == 0]
    nreal = min(n // 2, len(real))
    bogus = [(i, sc[0], sc[1]) for i, sc in classifications.items() if sc[0] == 1]
    nbogus = min(n - nreal, len(bogus))

    mc = real[:nreal] + bogus[:nbogus]
    fig, axes = plt.subplots(len(mc), 6, dpi=600, sharex='row')
    plt.subplots_adjust(hspace=0.1, wspace=0.02, left=0.2)
    zscale = ZScaleInterval()

    columns = ['science', 'template', 'difference', 'indiv_ptp', 'min_indiv', 'max_indiv']
    row_str_fmt = '{}: {}/{:.2f}'
    rows = [row_str_fmt.format(_mc[0], 'REAL' if _mc[1] else 'BOGUS', _mc[2]) for _mc in mc]

    for i, _mc in enumerate(mc):
        ax = axes[i]
        data_stamp = data_stamps[_mc[0]]
        for j in range(data_stamps.shape[-1]):
            layer = data_stamp[:, :, j]
            low, upp = zscale.get_limits(layer)
            layer_scaled = np.clip(layer, low, upp)
            ax[j].imshow(layer_scaled, cmap='gray')
            ax[j].get_xaxis().set_ticks([])
            ax[j].get_yaxis().set_ticks([])
            ax[j].tick_params(axis='both', which='both', length=0)
            if ax[j].is_first_col():
                ax[j].set_ylabel(rows[i], rotation=0, fontsize=6, labelpad=40)
            if ax[j].is_first_row():
                ax[j].set_title(columns[j], fontsize=3)

    plt.savefig(plot_filename)
