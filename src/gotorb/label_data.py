import argparse
import glob
import logging
import os
import uuid
from itertools import repeat
from multiprocessing import Pool, cpu_count

import astropy.units as u
import catsHTM  # should be the patched version
import numpy as np
import pandas as pd
import pympc
import tables
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.wcs import WCS
from decouple import config

from .skybot_query import query_skybot
from .utils import get_timestamp_str, gotophoto_connection, work_to_storage_filepath
from .utils import swap_dataloc

LOGGING_FORMAT = '%(asctime)s  %(levelname)-10s %(processName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

SCIENCE_IMAGE_EXT = 'IMAGE'
TEMPLATE_IMAGE_EXT = 'TEMPLATE'
DIFFERENCE_IMAGE_EXT = 'DIFFERENCE'
DIFFERENCE_PHOT_EXT = 'PHOTOMETRY_DIFF'

LABELS_FILENAME = 'data_labels.csv'
STAMPS_FILENAME = 'data_stamps.h5'
CATSHTM_DIR = config("CATSHTM_DIR")


def setup_logging(out_dir, logging_fn):
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(out_dir, logging_fn))
        fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logger.addHandler(fh)
        pympclogger = logging.getLogger('pympc')
        pympclogger.addHandler(fh)


def make(nimages, out_dir='.', mp_prov='skybot', search_rad=1.8, match_rad=1, max_mag=20.2,
         stamp_size=55, bogus_every=4, nproc=0, image_chunk=1500):
    """
    Creates a set of labelled data by using mpc-matches as real and random detections as bogus.

    Photometry and stamps are taken from the `DIFFERENCE` images in the `image` table of the gotophoto database.
    Results are stored in a directory within `out_dir` named using a timestamp. Stores a csv file of detections
    with labels and some metadata, as well as a `h5` file with an array of image stamps around each detection.
    Each stamp has 6 layers, which are [median science, template, difference, individual peak-to-peak,
    individual minimum, individual maximum].

    :param nimages: number of images to search for minor body cross-matches within
    :param out_dir: directory to write label src and stamps to
    :param mp_prov: MP search provider - is either "skybot" or "pympc". defaults to skybot
    :param search_rad: search radius in degrees around central ra,dec of an image for minor body positions
    :param match_rad: maximum cross-match radius in arcseconds between detections and minor body positions
    :param max_mag: maximum magnitude of minor bodies to cross match detections with
    :param stamp_size: size of image stamp to cutout around detections
    :param bogus_every: number of bogus detections to grab from every image (regardless of true mpc matches)
    :param nproc: amount of processes used for multiprocessed search, set to 1 turn off multiprocessing. 0 = #cpu
    :param image_chunk: splits the process of saving src to disk into chunks of this size, to prevent over-filling
                        memory
    :return:
    """

    timestamp = get_timestamp_str()
    out_dir = os.path.join(out_dir, 'data_{}'.format(timestamp))
    os.makedirs(out_dir)
    setup_logging(out_dir, 'data_{}.log'.format(timestamp))
    logger.info('output to be stored in {}'.format(out_dir))

    image_df_full = get_images(nimages)
    logger.info('### adding minor planet/random junk stamps ###')
    logger.info('nimages = {}'.format(nimages))
    logger.info('search_rad = {} degrees'.format(search_rad))
    logger.info('match_rad = {} arcsec'.format(match_rad))
    logger.info('max_mag = {} mag'.format(max_mag))
    logger.info('stamp_size = {} pixels'.format(stamp_size))
    logger.info('bogus_every = {}'.format(bogus_every))

    nproc = cpu_count() if nproc == 0 else nproc
    n_chunks = max(1, len(image_df_full) // image_chunk)

    logger.info('splitting {} images into {} chunks'.format(len(image_df_full), n_chunks))
    n = 0  # unique id to store for each src label
    for i, image_df in enumerate(np.array_split(image_df_full, n_chunks)):
        logger.info('processing image chunk {}/{}'.format(i + 1, n_chunks))

        if mp_prov == 'skybot':
            logger.info("using SkyBoT handler")
        if mp_prov == 'pympc':
            logger.info("using PyMPC offline search")

        if nproc > 1:
            # multiprocessing
            with Pool(nproc) as pool:
                labelled_stamps_list = pool.starmap(get_mpc_matches,
                                                    zip(image_df.iterrows(),
                                                        repeat(search_rad),
                                                        repeat(match_rad),
                                                        repeat(max_mag),
                                                        repeat(stamp_size),
                                                        repeat(mp_prov),
                                                        repeat(bogus_every))
                                                    )
        else:
            # single process
            labelled_stamps_list = []
            for irow in image_df.iterrows():
                labelled_stamps_list.append(get_mpc_matches(irow,
                                                            search_rad,
                                                            match_rad,
                                                            max_mag,
                                                            stamp_size,
                                                            mp_prov,
                                                            bogus_every,
                                                            chunk_size=50000)
                                            )
        logger.info('collating results')
        labelled_stamps_list = [item for sublist in labelled_stamps_list if sublist for item in sublist]

        logger.info('writing stamps and labels for chunk {}/{}'.format(i + 1, n_chunks))

        csv_file = os.path.join(out_dir, LABELS_FILENAME)
        df = pd.DataFrame()
        h5_file = os.path.join(out_dir, STAMPS_FILENAME)
        h5_out = tables.open_file(h5_file, mode='a')
        atom = tables.Float64Atom()
        if '/src' not in h5_out:
            array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, stamp_size, stamp_size, 6))
        else:
            array_c = h5_out.root.src
        for j, ls in enumerate(labelled_stamps_list):
            array_c.append(ls['stamps'][np.newaxis, :, :, :])
            del ls['stamps']
            ls['number'] = n
            df = df.append(ls, ignore_index=True, sort=True)
            n += 1
        h5_out.close()
        try:
            open(csv_file)
        except FileNotFoundError:
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', index=False, header=False)

        logger.info('chunk complete - {} stamps written'.format(len(labelled_stamps_list)))

    logger.info("DONE: generated {} mpc/random junk stamps in total".format(n))


def add_marshall_cands(data_dir, marshall_file, stamp_size=55, nproc=40, n_read=None, image_chunk=1500):
    """
    Create (or append) marshall candidates to a classifier-ready training set.

    :param data_dir: directory where model stamps and labels should be appended/made
    :param marshall_file: csv file of candidate ids and labels
    :param stamp_size: size of cutouts to generate
    :param nproc: number of processes to spawn
    :param n_read: number of images to read from image_tab
    :param image_chunk: chunk size to save in - needed for large files.
    :return: None
    """

    # if datadir doesn't exist, make it
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    logfiles = glob.glob(os.path.join(data_dir, "*.log"))

    if len(logfiles) != 0:
        logfilename = logfiles[0].split("/")[-1]
        setup_logging(data_dir, logfilename)
        logger.info('output to be stored in existing logfile {}'.format(data_dir))
    else:
        timestamp = get_timestamp_str()
        logfilename = "data_{}.log".format(timestamp)
        setup_logging(data_dir, logfilename)
        logger.info('output to be stored in new logfile {}'.format(data_dir))

    logger.info("### adding new stamps from marshall file ###")
    logger.info("marshall_file: {}".format(marshall_file))
    logger.info("stamp_size: {}".format(stamp_size))
    logger.info("n_read: {}".format(n_read))

    logger.info("ingesting marshall matches from {}".format(marshall_file))
    marshall_dets = pd.read_csv(marshall_file)
    cand_ids = marshall_dets.photometry_id
    label = marshall_dets.label

    # get relevant tables
    candidate_tab = get_candidate_info(cand_ids)
    candidate_tab["cand_id"] = cand_ids
    candidate_tab["label"] = label

    image_tab_full = id_to_images(cand_ids)

    # restrict number of images to read in
    if n_read is not None and n_read < len(image_tab_full):
        image_tab_full = image_tab_full.loc[:n_read, :]

    logger.info("tables processed, beginning stamp extraction with {} procs".format(nproc))
    nproc = cpu_count() if nproc == 0 else nproc
    n_chunks = max(1, len(image_tab_full) // image_chunk)

    logger.info('splitting {} images into {} chunks'.format(len(image_tab_full), n_chunks))
    for i, image_tab in enumerate(np.array_split(image_tab_full, n_chunks)):
        logger.info('processing image chunk {}/{}'.format(i + 1, n_chunks))
        # multiprocessing

        with Pool(nproc) as pool:
            marshall_stamps = pool.starmap(_get_marshall_matches,
                                           zip(image_tab.iterrows(), repeat(candidate_tab), repeat(stamp_size)))

        # convert list of lists to one list.
        marshall_stamps = [item for sublist in marshall_stamps if sublist for item in sublist]
        logger.info('writing stamps and labels for marshall chunk {}/{}'.format(i + 1, n_chunks))

        csv_file = os.path.join(data_dir, LABELS_FILENAME)
        df = pd.DataFrame()
        h5_file = os.path.join(data_dir, STAMPS_FILENAME)
        h5_out = tables.open_file(h5_file, mode='a')
        atom = tables.Float64Atom()
        if '/src' not in h5_out:
            array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, stamp_size, stamp_size, 6))
        else:
            array_c = h5_out.root.src

        n = 0

        for j, ls in enumerate(marshall_stamps):
            array_c.append(ls['stamps'][np.newaxis, :, :, :])
            del ls['stamps']
            ls['number'] = n
            df = df.append(ls, ignore_index=True, sort=True)
            n += 1
        h5_out.close()
        try:
            open(csv_file)
        except FileNotFoundError:
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', index=False, header=False)

    logger.info("DONE: {} marshall-selected stamps added in total".format(n))
    return None


def get_images(nimages):
    """
    Retrieve differenced images from gotophoto database.
    """
    query = ("select * from image "
             "where image_type = 'DIFFERENCE' "
             "and abs(dec_c) < 35 "
             "and quality < 128 "
             "and ncoadds >=3 "
             "limit %s")

    with gotophoto_connection() as conn:
        return pd.read_sql(query, conn, params=(nimages,))


def get_individual_images(relatedimageid):
    """
    Retrieve the individual SCIENCE images that made up the science image in a DIFFERENCE image
    """
    query = ("select * from image "
             "where image_type = 'SCIENCE' "
             "and relatedimage = %s")

    with gotophoto_connection() as conn:
        return pd.read_sql(query, conn, params=(relatedimageid,))


def get_mpc_matches(image_series, search_rad, match_rad, max_mag, stamp_size, mp_prov, every=0, chunk_size=0):
    """
    Perform a wide search for minor bodies over image FoV and cross-match them
    with detections in the image detections.

    For detections with successful cross-matches return cutouts of the
    science, template and difference images at those locations.
    We also grab the same number of random detections which are (very likely)
    bogus in the image. If `every` is given then we take this number of random "bogus"
    detections from every image always.
    """
    _, row = image_series
    ut = int(row['instrument'][-1])

    if mp_prov == 'pympc':
        mpc_search = pympc.minor_planet_check(row.ra_c, row.dec_c, row.jd - 2400000.5,
                                              search_rad * 3600., max_mag=max_mag, chunk_size=chunk_size)

        mpc_ras = [match[0][0] for match in mpc_search]
        mpc_decs = [match[0][1] for match in mpc_search]
        mpc_coo = SkyCoord(mpc_ras, mpc_decs, unit='deg')

    if mp_prov == 'skybot':
        mpc_search = query_skybot(row.ra_c, row.dec_c, search_rad, row.jd, maglim=max_mag)

        try:
            logger.info("found {} MPs in image".format(len(mpc_search)))
            mpc_ras = list(mpc_search['RA(h)'])
            mpc_decs = list(mpc_search['DE(deg)'])
            mpc_coo = SkyCoord(mpc_ras, mpc_decs, unit=(u.hourangle, u.deg))
        except TypeError:
            logger.warning("No MPs found!")
            return

    fits_filepath = work_to_storage_filepath(row.filepath)
    try:
        fits_file = fits.open(fits_filepath)
    except FileNotFoundError:
        try:
            fits_file = fits.open(swap_dataloc(fits_filepath))
        except FileNotFoundError:
            logger.warning("couldn't find {}".format(fits_filepath))
            return
    try:
        diff_phot_tbl = Table(fits_file[DIFFERENCE_PHOT_EXT].data)
    except KeyError:
        logger.warning("couldn't find {} extension in {}".format(DIFFERENCE_PHOT_EXT, fits_filepath))
        return
    if len(diff_phot_tbl) < 2:
        logger.info('(almost) empty table found in {}[{}]'.format(fits_filepath, DIFFERENCE_PHOT_EXT))
        return
    diff_coo = SkyCoord(diff_phot_tbl['ra'], diff_phot_tbl['dec'], unit='deg')

    try:
        idx, d2d, _ = mpc_coo.match_to_catalog_sky(diff_coo)
    except ValueError:
        logger.error('failed with {}'.format(fits_filepath))
        return
    # noinspection PyUnresolvedReferences
    matches = diff_phot_tbl[idx[(d2d <= match_rad * u.arcsec)]]
    if 'realbogus' not in matches.colnames:
        logger.warning('no realbogus column in difference photometry for {}'.format(fits_filepath))
        return
    # little safety net to avoid spuriously matching to RB=0 dets too often
    matches = matches[matches['realbogus'] > 0.01]
    logger.info('found {} mpc cross-matches'.format(len(matches)))

    results = []
    if not len(matches) and every == 0:
        return

    indiv_images_df = get_individual_images(row.relatedimage)
    logger.info('got {} individual science images'.format(len(indiv_images_df)))
    try:
        indiv_fits_files = [fits.open(work_to_storage_filepath(row.filepath)) for _, row in indiv_images_df.iterrows()]
    except FileNotFoundError:
        try:
            indiv_fits_files = [fits.open(swap_dataloc(work_to_storage_filepath(row.filepath))) for _, row in
                                indiv_images_df.iterrows()]
        except FileNotFoundError:
            logger.warning("couldn't find an individual file for {}".format(row.filepath))
            return

    # Store our cross-matches
    for match in matches:
        stamps = _get_stamps(fits_file, match['ra'], match['dec'], stamp_size, indiv_fits_files)
        if stamps is not None:
            res = dict(
                id=str(uuid.uuid4()),
                stamps=stamps,
                image_id=row['id'],
                x=match['x'],
                y=match['y'],
                ra=match['ra'],
                dec=match['dec'],
                mag=match['mag'],
                fwhm=match['fwhm'],
                realbogus=match['realbogus'],
                fits_filepath=fits_filepath,
                ut=ut,
                ncoadd=len(indiv_fits_files),
                label=1,  # REAL!
                metalabel='mp'
            )
            results.append(res)

    # if every == 0, we're done
    if every == 0:
        return results

    # Store an equal number of random, low real-bogus detections as bogus (we increase the search radius here,
    # because offline mpc positions can be inaccurate up to an arcminute, so we want to make sure we aren't grabbing
    # mpcs, which will be the main source of real objects in the src
    mask = np.ones(len(diff_phot_tbl))
    mask[idx] = 0
    unmatched_idx = np.where(mask == 1)[0]
    random_sample = min(len(unmatched_idx), len(matches) + every)
    nonmatches = diff_phot_tbl[np.random.choice(unmatched_idx, random_sample, replace=False)]
    nonmatches = nonmatches[nonmatches['realbogus'] < 0.6]
    for nonmatch in nonmatches:
        stamps = _get_stamps(fits_file, nonmatch['ra'], nonmatch['dec'], stamp_size, indiv_fits_files)
        if stamps is not None:
            res = dict(
                id=str(uuid.uuid4()),
                stamps=stamps,
                image_id=row['id'],
                x=nonmatch['x'],
                y=nonmatch['y'],
                ra=nonmatch['ra'],
                dec=nonmatch['dec'],
                mag = nonmatch['mag'],
                fwhm = nonmatch['fwhm'],
                realbogus=nonmatch['realbogus'],
                fits_filepath=fits_filepath,
                ut=ut,
                ncoadd=len(indiv_fits_files),
                label=0,  # BOGUS! (probably...)
                metalabel='randjunk'
            )
            results.append(res)
    return results


def _get_stamps(fits_file, ra, dec, size, indiv_fits_files=None, return_indiv_stamps=False):
    """
    Cut out science, template, and difference stamps around a detection at `ra,dec` from `fits_file`
    """

    def get_indiv_stamps():
        if not indiv_fits_files:
            return
        _indiv_stamps = np.zeros((size, size, len(indiv_fits_files)))
        for j, indiv_fits_file in enumerate(indiv_fits_files):
            try:
                im_wcs = WCS(indiv_fits_file[SCIENCE_IMAGE_EXT].header)
                stamp_coo = SkyCoord(ra, dec, unit='deg')
                indiv_cutout = Cutout2D(indiv_fits_file[SCIENCE_IMAGE_EXT].data, stamp_coo, (size, size),
                                        mode='partial', fill_value=1e-6, wcs=im_wcs).data
            except KeyError:
                logger.warning("extension {} missing from {}".format(image_ext, fits_file))
                return

            except NoOverlapError:
                logger.warning("requested stamp(s) are off edge of image")
                return

            _indiv_stamps[:, :, j] = indiv_cutout
        return _indiv_stamps

    # layers are: science, template, difference, peak-to-peak individual, min individual, max individual
    stamps = np.zeros((size, size, 6))
    for i, image_ext in enumerate([SCIENCE_IMAGE_EXT, TEMPLATE_IMAGE_EXT, DIFFERENCE_IMAGE_EXT]):
        try:
            im_wcs = WCS(fits_file[image_ext].header)
            stamp_coo = SkyCoord(ra, dec, unit='deg')
            cutout = Cutout2D(fits_file[image_ext].data, stamp_coo, (size, size),
                              mode='partial', fill_value=1e-6, wcs=im_wcs, ).data
        except KeyError:
            logger.warning("extension {} missing from {}".format(image_ext, fits_file))
            return
        except NoOverlapError:
            logger.warning("requested stamp is off edge of image")
            return
        stamps[:, :, i] = cutout

    indiv_stamps = get_indiv_stamps()

    # modify return signature to support multiarg.
    if (indiv_stamps is None) and ~return_indiv_stamps:
        return stamps

    if (indiv_stamps is None) and return_indiv_stamps:
        return stamps, None

    # Add additional frames that capture statistics from the individual science images
    stamps[:, :, 3] = np.ptp(np.atleast_3d(indiv_stamps), axis=2)
    stamps[:, :, 4] = np.min(np.atleast_3d(indiv_stamps), axis=2)
    stamps[:, :, 5] = np.max(np.atleast_3d(indiv_stamps), axis=2)

    if return_indiv_stamps:
        return stamps, indiv_stamps
    else:
        return stamps


def get_candidate_info(candid):
    """
        Retrieve candidate table for specific candidate ids

        :param candid: candidate id
        :return: pandas dataframe of images
        """
    candtuple = tuple(candid)
    query = '''select * from candidate where id in %s'''

    with gotophoto_connection() as conn:
        return pd.read_sql(query, conn, params=(candtuple,))


def id_to_images(imageid):
    """
    Retrieve unique images from gotophoto database for specific candidate ids

    :param imageid: list of image ids
    :return: pandas dataframe of images
    """
    # force to be Python int

    imageidtuple = tuple(imageid)
    query = '''select * from image where id in
            (select image_id from candidate where
            id in %s) and ncoadds > 0 and quality<128
            '''

    with gotophoto_connection() as conn:
        return pd.read_sql(query, conn, params=(imageidtuple,))


def _get_marshall_matches(imageseries, candidate_tab, stamp_size):
    _, row = imageseries
    ut = int(row['instrument'][-1])
    fits_filepath = work_to_storage_filepath(row.filepath)
    try:
        fits_file = fits.open(fits_filepath)
    except FileNotFoundError:
        try:
            fits_file = fits.open(swap_dataloc(fits_filepath))
        except FileNotFoundError:
            logger.info("file not found!")
            return

    indiv_images_df = get_individual_images(row.relatedimage)
    logger.info('got {} individual science images'.format(len(indiv_images_df)))

    try:
        indiv_fits_files = [fits.open(work_to_storage_filepath(row.filepath)) for _, row in indiv_images_df.iterrows()]
    except FileNotFoundError:
        try:
            indiv_fits_files = [fits.open(swap_dataloc(work_to_storage_filepath(row.filepath)))
                                for _, row in indiv_images_df.iterrows()]

        except FileNotFoundError:
            logger.info("couldn't find an individual file for {}".format(row.filepath))
            return

    # grab candidates in this image
    cands = candidate_tab["image_id"] == row.id
    matches = candidate_tab[cands]

    imres = []

    logger.info("found {} marshall dets in frame".format(len(matches)))
    for match in matches.itertuples():
        stamps = _get_stamps(fits_file, match.ra, match.dec, stamp_size, indiv_fits_files)
        aug_label = match.label
        if stamps is not None:
            res = dict(
                id=str(uuid.uuid4()),
                stamps=stamps,
                image_id=row.id,
                x=match.x,
                y=match.y,
                ra=match.ra,
                dec=match.dec,
                mag=match.mag,
                fwhm=match.fwhm,
                realbogus=match.realbogus,
                fits_filepath=fits_filepath,
                ut=ut,
                ncoadd=len(indiv_fits_files),
                label=int(aug_label),  # this might not always be bogus, so reference it to source file
                metalabel='marshall'
            )

            imres.append(res)

    return imres


def add_synthetic_transients(nimages, out_dir='.', match_rad=1, search_rad=1.8, min_mag=16, max_mag=19.5, glx_search_scale=600,
                        glx_jitter=7, stamp_size=55, nproc=20, resid_every=1, image_chunk=1500):
    """
    Routine for adding synthetic transients to a dataset. Calls make_sim_transient using the same multiproc wrapper as
    label_data.make().

    :param nimages: number of images to ingest
    :param out_dir: output directory for the stamps - if doesn't exist will be made
    :param match_rad: radius for MP cross-matching - should be ~1" for optimal results
    :param search_rad: size of radius for initial MP crossmatch - should be ~ size of image
    :param min_mag: bright limit for minor planets - remove detections that completely overpower host galaxy
    :param max_mag: faint limit for minor planets - should be brighter than normal to ensure decent SNR transients
    :param glx_search_scale: maximum distance from each minor planet to choose a galaxy from - coherency of PSF
    :param glx_jitter: uniform scatter to apply to galaxy stamp to simulate offset transients
    :param stamp_size: size of stamps to extract
    :param nproc: number of processes to use - best results at 32
    :param resid_every: number of residuals per source to inject
    :param image_chunk: size of chunks to write images in
    :return: None
    """

    # if datadir doesn't exist, make it
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    logfiles = glob.glob(os.path.join(out_dir, "*.log"))

    if len(logfiles) != 0:
        logfilename = logfiles[0].split("/")[-1]
        setup_logging(out_dir, logfilename)
        logger.info('output to be stored in existing logfile {}'.format(out_dir))
    else:
        timestamp = get_timestamp_str()
        logfilename = "data_{}.log".format(timestamp)
        setup_logging(out_dir, logfilename)
        logger.info('output to be stored in new logfile {}'.format(out_dir))

    # only put data-related params
    logger.info("### adding synthetic transient/galaxy residual stamps ###")
    logger.info("nimages: {}".format(nimages))
    logger.info("match_rad: {}".format(match_rad))
    logger.info("search_rad: {}".format(search_rad))
    logger.info("min_mag: {}".format(min_mag))
    logger.info("max_mag: {}".format(max_mag))
    logger.info("glx_search_scale: {}".format(glx_search_scale))
    logger.info("glx_jitter: {}".format(glx_jitter))
    logger.info("stamp_size: {}".format(stamp_size))
    logger.info("resid_every: {}".format(resid_every))

    n = 0  # unique id for each
    image_df_full = get_images(nimages)
    nproc = cpu_count() if nproc == 0 else nproc
    n_chunks = max(1, len(image_df_full) // image_chunk)

    logger.info('splitting {} images into {} chunks'.format(len(image_df_full), n_chunks))

    for i, image_df in enumerate(np.array_split(image_df_full, n_chunks)):
        logger.info('processing image chunk {}/{}'.format(i + 1, n_chunks))

        with Pool(nproc) as pool:
            labelled_stamps_list = pool.starmap(_make_sim_transient,
                                                zip(image_df.iterrows(),
                                                    repeat(match_rad),
                                                    repeat(search_rad),
                                                    repeat(min_mag),
                                                    repeat(max_mag),
                                                    repeat(glx_search_scale),
                                                    repeat(glx_jitter),
                                                    repeat(stamp_size),
                                                    repeat(resid_every))
                                                )

        labelled_stamps_list = [item for sublist in labelled_stamps_list if sublist for item in sublist]

        logger.info('writing stamps and labels for chunk {}/{}'.format(i + 1, n_chunks))
        logger.info('{} stamps in chunk'.format(len(labelled_stamps_list)))
        csv_file = os.path.join(out_dir, LABELS_FILENAME)
        df = pd.DataFrame()
        h5_file = os.path.join(out_dir, STAMPS_FILENAME)
        h5_out = tables.open_file(h5_file, mode='a')
        atom = tables.Float64Atom()
        if '/src' not in h5_out:
            array_c = h5_out.create_earray(h5_out.root, 'src', atom, (0, stamp_size, stamp_size, 6))
        else:
            array_c = h5_out.root.src

        for j, ls in enumerate(labelled_stamps_list):
            array_c.append(ls['stamps'][np.newaxis, :, :, :])
            del ls['stamps']
            ls['number'] = n
            df = df.append(ls, ignore_index=True)
            n += 1
        h5_out.close()
        try:
            open(csv_file)
        except FileNotFoundError:
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', index=False, header=False)

        logger.info("DONE: {} transient/residual stamps generated in total".format(n))


def _make_sim_transient(inputrow, match_rad=2, search_rad=1.8, min_mag=16, max_mag=20.2, glx_search_scale=600,
                        glx_jitter=5, stamp_size=55, resid_every=0, rng_state=42):
    """
    internal routine to generate transients

    :param inputrow: row of dataframe (from label_data.get_images()) generated by iterrows()
    :param match_rad: radius for MP cross-matching - should be ~1" for optimal results
    :param search_rad: size of radius for initial MP crossmatch - should be ~ size of image
    :param min_mag: bright limit for minor planets - remove detections that completely overpower host galaxy
    :param max_mag: faint limit for minor planets - should be brighter than normal to ensure decent SNR transients
    :param glx_search_scale: maximum distance from each minor planet to choose a galaxy from - coherency of PSF
    :param glx_jitter: uniform scatter to apply to galaxy stamp to simulate offset transients
    :param stamp_size: size of stamps to extract
    :param resid_every: number of galaxy residuals
    :param rng_state: testing seed for repeatability
    :return:
    """

    np.random.seed(rng_state)
    _, row = inputrow

    ut = int(row['instrument'][-1])

    fits_filepath = work_to_storage_filepath(row.filepath)
    try:
        fits_file = fits.open(fits_filepath)
    except FileNotFoundError:
        try:
            fits_file = fits.open(swap_dataloc(fits_filepath))
        except FileNotFoundError:
            logger.warning("corresponding file not found")
            return

    indiv_images_df = get_individual_images(row.relatedimage)
    try:
        indiv_fits_files = [fits.open(work_to_storage_filepath(row.filepath)) for _, row in indiv_images_df.iterrows()]
    except FileNotFoundError:
        try:
            indiv_fits_files = [fits.open(swap_dataloc(work_to_storage_filepath(row.filepath))) for _, row in
                                indiv_images_df.iterrows()]
        except FileNotFoundError:
            logger.warning("Individual FITS files missing: {}".format(work_to_storage_filepath(row.filepath)))
            return

    mpc_search = query_skybot(row.ra_c, row.dec_c, search_rad, row.jd, maglim=max_mag)

    try:
        logger.info("found {} MPs in image".format(len(mpc_search)))
        mpc_ras = list(mpc_search['RA(h)'])
        mpc_decs = list(mpc_search['DE(deg)'])
        mpc_coo = SkyCoord(mpc_ras, mpc_decs, unit=(u.hourangle, u.deg))
    except TypeError:
        logger.warning("no MPs found!")
        return

    try:
        diff_phot_tbl = Table(fits_file[DIFFERENCE_PHOT_EXT].data)
    except KeyError:
        logger.warning("couldn't find {} extension in {}".format(DIFFERENCE_PHOT_EXT, fits_filepath))
        return
    if len(diff_phot_tbl) < 2:
        logger.info('(almost) empty table found in {}[{}]'.format(fits_filepath, DIFFERENCE_PHOT_EXT))
        return
    diff_coo = SkyCoord(diff_phot_tbl['ra'], diff_phot_tbl['dec'], unit='deg')

    try:
        idx, d2d, _ = mpc_coo.match_to_catalog_sky(diff_coo)
    except ValueError:
        logger.error('failed with {}'.format(fits_filepath))
        return
    # noinspection PyUnresolvedReferences
    matches = diff_phot_tbl[idx[(d2d <= match_rad * u.arcsec)]]
    if 'realbogus' not in matches.colnames:
        logger.warning('no realbogus column in difference photometry for {}'.format(fits_filepath))
        return
    # little safety net to avoid spuriously matching to RB=0 dets too often
    matches = matches[matches['realbogus'] > 0.01]
    logger.info('found {} mpc cross-matches'.format(len(matches)))

    if matches is None:
        logger.warning("no minor planets found!")
        return

    # optimise - query GLADE once per image to save time
    glade_data, glade_cols, _ = catsHTM.cone_search("GLADE", (np.pi / 180) * row.ra_c, (np.pi / 180) * row.dec_c,
                                                    search_rad * 3600,
                                                    catalogs_dir=CATSHTM_DIR)
    glade_tab = Table(glade_data, names=glade_cols)
    glade_tab = glade_tab[np.isfinite(glade_tab["B"])]  # clean the NaNs to avoid warnings
    glade_tab = glade_tab[glade_tab["B"] < 18]  # maglim cut

    logger.info("{} GLADE galaxies in frame".format(len(glade_tab)))

    results = []
    for match in matches:
        # remove detections that are too bright and saturate the detector.
        if match["mag"] < min_mag:
            logger.warning("MP exceeds bright threshold")
            continue

        glade_coords = SkyCoord(glade_tab["RA"], glade_tab["Dec"], unit=u.rad)  # keep in to update table on iter
        mp_coords = SkyCoord(match["ra"], match["dec"], unit=u.deg)
        nn_dists = mp_coords.separation(glade_coords).to(u.arcsec).value
        temp_tab = glade_tab[nn_dists < glx_search_scale]

        if len(temp_tab) == 0:
            logger.warning("No GLADE galaxies within {} arcsec".format(glx_search_scale))
            continue

        glx_sel = temp_tab[np.argmin(temp_tab["B"])]  # greedily select bright galaxies

        try:
            glx_stamp, glx_indiv = _get_stamps(fits_file,
                                    (180 / np.pi) * glx_sel["RA"] + np.random.uniform(-glx_jitter, glx_jitter) / 3600,
                                    (180 / np.pi) * glx_sel["Dec"] + np.random.uniform(-glx_jitter, glx_jitter) / 3600,
                                    stamp_size,
                                    indiv_fits_files, return_indiv_stamps=True)
        except NoOverlapError:
            logger.warning("galaxy located off edge of stamp")
            continue

        # HACK need to work out what the problem is with this...
        except (TypeError, ValueError):
            logger.warning("problem with stamp extraction")
            continue

        try:
            mp_stamp, mp_indiv = _get_stamps(fits_file, match["ra"], match["dec"], stamp_size, indiv_fits_files,
                                             return_indiv_stamps=True)

        # HACK similarly here
        except (TypeError, ValueError):
            logger.warning("Problem with stamp extraction")
            continue

        comb_stamp = mp_stamp + glx_stamp
        assert mp_indiv.shape == glx_indiv.shape # crash if the stamps aren't equiv
        indiv_img_comb = mp_indiv + glx_indiv

        # need to respect addition of p2p stamps correctly.
        comb_stamp[:, :, 3] = np.ptp(np.atleast_3d(indiv_img_comb), axis=2) # peak to peak
        comb_stamp[:, :, 4] = np.min(np.atleast_3d(indiv_img_comb), axis=2) # min
        comb_stamp[:, :, 5] = np.max(np.atleast_3d(indiv_img_comb), axis=2) # max

        if comb_stamp is not None:
            res = dict(
                id=str(uuid.uuid4()),
                stamps=comb_stamp,
                image_id=row['id'],
                x=match['x'],
                y=match['y'],
                ra=match['ra'],
                dec=match['dec'],
                mag=match['mag'],
                fwhm=match['fwhm'],
                realbogus=match['realbogus'],
                fits_filepath=fits_filepath,
                ut=ut,
                ncoadd=len(indiv_fits_files),
                label=1,  # synthetic real transient
                metalabel='syntransient'
            )
            results.append(res)

        # avoid duplicate seed galaxies
        glade_tab.remove_row(
            np.int(np.argwhere((glade_tab["RA"] == glx_sel["RA"]) & (glade_tab["Dec"] == glx_sel["Dec"]))))

    if len(glade_tab) == 0:
        logger.info("ran out of galaxies")
        return results

    if resid_every == 0:
        return results

    glade_tab.sort("B") # sort in ascending order (i.e. bright end first)
    temp_coo = SkyCoord((180/np.pi) * glade_tab["RA"], (180/np.pi) * glade_tab["Dec"], unit=u.deg)

    ndets = int(min(resid_every*len(matches), len(matches))) # make sure we never exceed the number of matches
    im_wcs = WCS(fits_file[1].header) # parse WCS for computing detector position
    for idx, c in enumerate(temp_coo[:ndets]):
        try:
            glx_resid_stamp = _get_stamps(fits_file,
                                          c.ra.value + np.random.uniform(-glx_jitter/2, glx_jitter/2) / 3600,
                                          c.dec.value + np.random.uniform(-glx_jitter/2, glx_jitter/2) / 3600,
                                          stamp_size, indiv_fits_files)
        except NoOverlapError:
            logger.warning("galaxy located off edge of stamp")
            continue

        # the top except should catch any problems with this.
        xdet, ydet = im_wcs.all_world2pix([c.ra.value], [c.dec.value], 0)

        if glx_resid_stamp is not None:
            res = dict(
                id=str(uuid.uuid4()),
                stamps=glx_resid_stamp,
                image_id=row['id'],
                x=xdet[0],
                y=ydet[0],
                ra=c.ra.value,
                dec=c.dec.value,
                mag=glade_tab["B"][idx],
                fwhm=99.99, # this is ill-defined here, since we're not using detections
                realbogus=99.99, # similarly. b
                fits_filepath=fits_filepath,
                ut=ut,
                ncoadd=len(indiv_fits_files),
                label=0,  # galaxy residual
                metalabel='glxresid'
            )
            results.append(res)

    logger.info("{} transients and residuals added".format(len(results)))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nimages', type=int, default=400,
                        help='number of images to search for minor body cross-matches within')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='directory to write label src and stamps to')
    parser.add_argument('--search-rad', type=float, default=1.8,
                        help='search radius in degrees around central ra,dec of an image for minor body positions')
    parser.add_argument('--match-rad', type=float, default=4.0,
                        help='maximum cross-match radius in arcseconds between detections and minor body positions')
    parser.add_argument('--max-mag', type=float, default=20.2,
                        help='maximum magnitude of minor bodies to cross match detections with')
    parser.add_argument('--stamp-size', type=int, default=63,
                        help='size of image stamp to cutout around detections')
    parser.add_argument('--bogus-every', type=int, default=3,
                        help='number of bogus detections to grab from every image (regardless of true mpc matches)')
    parser.add_argument('--nproc', type=int, default=0,
                        help='amount of processes used for multiprocessed search, set to 1 turn off multiprocessing.'
                             'default (0) will use #cpu')
    parser.add_argument('--image-chunk', type=int, default=1000,
                        help='splits the process of saving src to disk into chunks of this size, to prevent '
                             'over-filling memory')
    args = parser.parse_args()

    make(**vars(args))
