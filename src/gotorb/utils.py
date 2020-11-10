from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import tables
from astropy.io import fits
from decouple import config
from scipy.stats import truncnorm

from src.gotorb.label_data import get_individual_images, _get_stamps


def get_timestamp_str():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

@contextmanager
def gotophoto_connection():
    kwargs = dict(
        dbname=config('GOTOPHOTO_DB_NAME'),
        host=config('GOTOPHOTO_DB_HOST'),
        port=config('GOTOPHOTO_DB_PORT'),
        user=config('GOTOPHOTO_DB_USER'),
    )

    with psycopg2.connect(**kwargs) as conn:
        yield conn
    conn.close()


def work_to_storage_filepath(filepath):
    return filepath.replace(config('GOTOPHOTO_WORKDIR'), config('GOTOPHOTO_STORAGEDIR')).replace('reduced', 'final')


def rb_colourmap(rbval):
    """
    Convenience function to return a HTML-compatible colour for a real-bogus score
    Factored out of visualise.py

    :param rbval: realbogus score, range (0,1)
    :return: colour
    """
    if rbval > 0.75:
        color = "green"
    elif rbval > 0.5:
        color = "gold"
    elif rbval > 0.25:
        color = "orange"
    else:
        color = "red"

    return color

def silverman_bw_estimator(posterior):
    """
    Estimates the optimal bandwidth for KDE smoothing of data using the Silverman bandwidth estimator.

    :param posterior: raw posterior samples
    :return: sigma-value to use with a KDE implementation.
    """
    return 1.06*np.std(posterior, ddof=1)*len(posterior)**(-1/5)

def compute_score_histogram(posterior, gridpoints=1000, smooth_method='silverman'):
    """
    Computes a real-bogus score histogram with truncated normal distribution KDE - this more accurately conserves
    probability density at boundary areas

    :param posterior: Set of posterior samples from dropout_pred
    :param gridpoints: Resolution of internal sampling grid
    :param smooth_method: "silverman" uses the Silverman rule, float sets fixed bandwidth, callable uses a user-defined
    rule for bandwidth
    :return: uniform grid on range (0,1) and corresponding posterior values
    """
    eval_range = np.linspace(0, 1, gridpoints)

    if smooth_method == 'silverman':
        sigma = silverman_bw_estimator(posterior)
    elif callable(smooth_method):
        sigma = smooth_method(posterior)
    elif type(smooth_method) == float:
        sigma = smooth_method
    else:
        raise ValueError("Invalid smooth method specified.")

    output_kde = []
    for p in posterior:
        mu = p
        gendist = truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
        output_kde.append(gendist.pdf(eval_range))

    return eval_range, np.sum(output_kde, axis=0)

def write_to_h5_file(h5_file, stamps):
    """
    Wrapper for writing stamps out to a `.h5` file.

    :param h5_file: location of existing h5 file, or path to generate a new file
    :param stamps: numpy array of stamps, shape [batch, x, y, channel]
    :return:
    """

    h5_out = tables.open_file(h5_file, mode='a')
    atom = tables.Float64Atom()
    if '/src' not in h5_out:
        array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, *np.shape(stamps[0, :, :, :])))
    else:
        array_c = h5_out.root.src

    array_c.append(stamps)
    h5_out.close()
    return

def swap_dataloc(filepath):
    """
    Convenience function to swap network mount locations for filepaths queried from the database.
    Should prevent FileNotFound errors until paths are static, at least over the data relevant for this project.

    :param filepath: input path
    :return: filepath: output path, now swapped
    """
    if "gotodata2" in filepath:
        return filepath.replace("gotodata2", "gotodata3")
    if "gotodata3" in filepath:
        return filepath.replace("gotodata3", "gotodata2")


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