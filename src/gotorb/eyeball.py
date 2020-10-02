import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from hickle import hickle
from matplotlib.widgets import Button, TextBox
from tensorflow import keras

from .classifier import MODELS_FILENAME, META_FILENAME
from .label_data import (LABELS_FILENAME, STAMPS_FILENAME, get_individual_images, _get_stamps)
from .utils import gotophoto_connection, work_to_storage_filepath

LABELS = {
    -1: 'unknown',
    0: 'BOGUS',
    1: 'REAL',
}

LAYERS = {
    0: 'science',
    1: 'template',
    2: 'difference',
    3: 'indiv_ptp',
    4: 'min_indiv',
    5: 'max_indiv',
}


def show(model_dir='.', label_class=None, hide_scores=False):
    """
    Show interactive plots of detections, labels and classifier scores.

    :param model_dir: typically a `model_YYMMDD-HHMMSS` directory containing `model.h5` and `meta.hkl`
                      files from a trained model. directly above this must be a `data_YYMMDD-HHMMSS`
                      directory containing `data_labels.csv` and `data_stamps.h5` files.
    :param label_class: show only objects that are labelled as real (1) or bogus (0). if None, show both
    :param hide_scores: whether to hide the realbogus scores from the classifiers - potentially useful
                        for a human eyeballing scheme.
    :return:
    """
    data_labels = pd.read_csv(os.path.join(model_dir, '..', LABELS_FILENAME), index_col=0)
    f = tables.open_file(os.path.join(model_dir, '..', STAMPS_FILENAME), mode='r')
    data_stamps = f.root.src

    model_filename = os.path.join(model_dir, MODELS_FILENAME)
    try:
        model = keras.models.load_model(model_filename)
    except OSError:
        raise("CNN model not found at {}".format(model_filename))
    meta = hickle.load(os.path.join(model_dir, META_FILENAME))

    layers = meta['layers']
    if layers is None:
        layers = np.arange(len(LAYERS))

    stampsize = np.shape(meta['x_test'][0])[0]

    fig, axes = plt.subplots(1, len(layers), figsize=(12, 6), sharex='all', sharey='all')

    class Index:
        data_label, data_stamp = get_data(data_labels, data_stamps, label_class)
        scale = 'zscale'

        def plot(self):
            for i, layer in enumerate(layers):
                title = LAYERS[layer]
                axes[i].set_title(title)
                data = self.data_stamp[:, :, layer]
                if self.scale == 'zscale':
                    zscale = ZScaleInterval()
                    low, upp = zscale.get_limits(data)
                    data = np.clip(data, low, upp)
                elif isinstance(self.scale, (float, int)):
                    difftohundred = (100. - self.scale)
                    lowpc = np.nanpercentile(data, difftohundred / 2.)
                    upppc = np.nanpercentile(data, self.scale + difftohundred / 2.)
                    data = np.clip(data, lowpc, upppc)
                axes[i].imshow(data, cmap='gray')
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)
                plt.tight_layout()
            suptitle = "{:}: UT{:d} {:.2f},{:.2f}. ncoadd = {:}.".format(
                self.data_label.index,
                int(self.data_label.ut),
                self.data_label.x,
                self.data_label.y,
                self.data_label.ncoadd,
            )
            if not hide_scores:
                suptitle += "\nRF: realbogus={:.2f}".format(self.data_label.realbogus)
                norm_data_stamp = np.array(keras.backend.l2_normalize(
                    self.data_stamp[np.newaxis, :, :, layers], axis=(1, 2))
                )
                cnn_score = model.predict(norm_data_stamp)[0][0]
                suptitle += "\nCNN: realbogus={:.2f}".format(cnn_score)
                suptitle += "\nLabel: {}".format(LABELS[self.data_label.label])
            fig.suptitle(suptitle)
            plt.draw()

        def zscale(self, event):
            self.scale = 'zscale'
            self.plot()

        def nnpf(self, event):
            self.scale = 99.5
            self.plot()

        def nf(self, event):
            self.scale = 95
            self.plot()

        def plot_specific(self, text):
            self.data_label, self.data_stamp = get_data(data_labels, data_stamps, idx=int(text))
            self.plot()

        def plot_specific_candidate(self, text):
            res = get_candidate_data(int(text), stampsize)
            if res is not None:
                self.data_label, self.data_stamp = res
                self.plot()

        def press(self, event):
            if event.key == 'r':
                vote = 'real'
            elif event.key == 'u':
                vote = 'unsure'
            elif event.key == 'b':
                vote = 'bogus'
            else:
                return
            #print("{} marked as {}".format(int(self.data_label.index), vote))
            self.data_label, self.data_stamp = get_data(data_labels, data_stamps)
            self.plot()

    callback = Index()
    callback.plot()
    fig.canvas.mpl_connect('key_press_event', callback.press)

    # scaling buttons
    zscaleax = plt.axes([0.05, 0.15, 0.1, 0.04])
    zscalebutton = Button(zscaleax, 'zscale', hovercolor='0.975')
    zscalebutton.on_clicked(callback.zscale)

    nnpfax = plt.axes([0.15, 0.15, 0.1, 0.04])
    nnpfbutton = Button(nnpfax, '99.5%', hovercolor='0.975')
    nnpfbutton.on_clicked(callback.nnpf)

    nfax = plt.axes([0.25, 0.15, 0.1, 0.04])
    nfbutton = Button(nfax, '95%', hovercolor='0.975')
    nfbutton.on_clicked(callback.nf)

    axbox = plt.axes([0.15, 0.10, 0.2, 0.05])
    text_box = TextBox(axbox, 'show data_labels index: ', initial='')
    text_box.on_submit(callback.plot_specific)

    axbox = plt.axes([0.15, 0.05, 0.2, 0.05])
    text_box = TextBox(axbox, 'show candidate_id: ', initial='')
    text_box.on_submit(callback.plot_specific_candidate)

    # info box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'press key to classify:',
        'r = real',
        'u = unsure',
        'b = bogus',
    ))
    plt.text(0.95, 0.1, textstr, fontsize=16,
             verticalalignment='bottom',
             horizontalalignment='right',
             transform=fig.transFigure,
             bbox=props)

    plt.show()
    f.close()


def get_data(labels, stamps, idx=None, label_class=None):
    if idx is not None:
        return labels.iloc[idx], stamps[idx]
    if label_class:
        idxs = np.where(labels['label'] == label_class)[0]
    else:
        idxs = np.arange(len(labels))
    i = np.random.choice(idxs)
    return labels.iloc[i], stamps[i]


def get_candidate_data(candidateid, stampsize):
    print('creating stamps for candidate id {}'.format(candidateid))
    query = ("select * from candidate join image on candidate.image_id = image.id "
             "where candidate.id = %s")
    with gotophoto_connection() as conn:
        row = pd.read_sql(query, conn, params=(candidateid,))
        if len(row) != 1:
            print("Got {} results for candidate id {}".format(len(row), candidateid))
            return
    row = row.iloc[0]
    fits_filepath = work_to_storage_filepath(row.filepath)
    try:
        fits_file = fits.open(fits_filepath)
    except FileNotFoundError:
        try:
            fits_file = fits.open(fits_filepath.replace("gotodata2", "gotodata3"))
        except FileNotFoundError:
            print("couldn't find {}".format(fits_filepath))
            return

    indiv_images_df = get_individual_images(row.relatedimage)
    try:
        indiv_fits_files = [fits.open(work_to_storage_filepath(row.filepath)) for _, row in indiv_images_df.iterrows()]
    except FileNotFoundError:
        try:
            indiv_fits_files = [fits.open(work_to_storage_filepath(row.filepath).replace("gotodata2", "gotodata3"))
                                for _, row in indiv_images_df.iterrows()]
        except FileNotFoundError:
            print("couldn't find an individual file for {}".format(row.filepath))
            return

    stamps = _get_stamps(fits_file, row.ra, row.dec, stampsize, indiv_fits_files)
    label = pd.Series(dict(
        image_id=row.image_id,
        x=row.x,
        y=row.y,
        ra=row.ra,
        dec=row.dec,
        mag=row.mag,
        fwhm=row.fwhm,
        index=candidateid,
        realbogus=row.realbogus,
        fits_filepath=fits_filepath,
        ut=int(row.instrument[-1]),
        ncoadd=len(indiv_fits_files),
        label=-1  # unknown
    ))
    return label, stamps
