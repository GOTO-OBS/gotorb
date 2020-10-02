import os
from itertools import repeat
from multiprocessing import Pool

import astropy.units as u
import numpy as np
import pandas as pd
import tables
import tqdm
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from .classifier import get_data_stamps, get_data_labels
from .label_data import swap_dataloc, LABELS_FILENAME, STAMPS_FILENAME
from .skybot_query import query_skybot
from .utils import write_to_h5_file


def clean_varstars(data_dir, refcat_fitsloc, match_rad):
    """
    Cross-match with ATLAS Var cat to find variable stars in a dataset. Useful as ~4% of marshall junk corresponds to
    variable stars

    :param data_dir: directory containing the datastamps and labels
    :param match_rad: cross-matching radius
    :param refcat_fitsloc: location of the ATLAS Variable Star Catalogue in FITS format
    :return: None
    """

    data_stamps_path = os.path.join(data_dir, STAMPS_FILENAME)
    data_labels_path = os.path.join(data_dir, LABELS_FILENAME)

    # load table of labels
    data_labels = pd.read_csv(data_labels_path)

    # load ATLAS Variable cat from FITS file - catalogue 'small' so *relatively* fast for batch queries like this
    # OPT replace with catsHTM if this proves to be a bottleneck
    print("loading ATLAS refcat from {}".format(refcat_fitsloc))
    atlasvarcat = Table(fits.getdata(refcat_fitsloc))

    print("Cross-matching candidate and source coordinates")
    cand_coords = SkyCoord(ra=data_labels.ra.values, dec=data_labels.dec.values, unit=(u.deg, u.deg))
    cat_coords = SkyCoord(ra=atlasvarcat["ra"], dec=atlasvarcat["dec"], unit=(u.deg, u.deg))

    # Use built in astropy x-match utility - fast on tables this size.
    idx, d2d, d3d = cand_coords.match_to_catalog_3d(cat_coords)
    var_mask = (d2d < match_rad * u.arcsec)
    print("Contamination: {:0.2f}%".format(100 * np.sum(var_mask) / len(cand_coords)))

    print("Loading stamps and labels from {}".format(data_dir))
    data_container = tables.open_file(data_stamps_path, mode='r')
    data_stamps = np.array(data_container.root.src)[~var_mask, :, :, :]
    data_container.close()

    new_stamps = data_stamps
    new_labels = data_labels.iloc[~var_mask]
    print("Writing stamps and labels to current directory")
    new_labels.to_csv(os.path.join(data_dir, "varcleaned_" + LABELS_FILENAME), index=False)
    h5_file = os.path.join(data_dir, "varcleaned_" + STAMPS_FILENAME)
    write_to_h5_file(h5_file, new_stamps)

    print("Variable star cleaning completed!")
    return

def clean_cosmics(data_dir, xmatch_dist, nproc):
    """
    Removes cosmic rays from a dataset by enforcing that the number of individual detections associated with each
    source not be 1. Logic here: 0 detections could be a source detected in median but not individual images
    2-3 dets: source detected
    1 det: likely a CR, appearing in only one image

    :param data_dir: directory containing the datastamps and labels
    :param xmatch_dist: threshold at which to associate sources - in arcsec
    :param nproc: number of processes to use
    :return: None
    """
    print("Reading in stamps and labels")
    data_labels = get_data_labels(os.path.join(data_dir, LABELS_FILENAME))
    data_stamps = get_data_stamps(os.path.join(data_dir, STAMPS_FILENAME))

    print("Beginning cosmic ray identification")
    with Pool(nproc) as pool:
        results = pool.starmap(_check_cosmics, zip(data_labels.iterrows(), repeat(xmatch_dist)))

    print("Cosmic ray identification complete:")
    res_labels = np.array(results)[:, 1]
    for frametype, name in zip([0, 1, 99], ["cosmics", "clean", "unchecked"]):
        print("{}: {} of {}".format(name, np.sum(res_labels == frametype), len(res_labels)))

    # Only want to reject bogus-labelled detections - chances of MP/transient stamp being CR are minimal.
    cosmics_mask = (data_labels["label"] == 0) & (res_labels == 0)

    print("Writing new stamps and labels")
    new_tab = data_labels[~cosmics_mask]
    new_tab.to_csv(os.path.join(data_dir, "cosmiccleaned_" + LABELS_FILENAME), index=False)

    h5_file = os.path.join(data_dir, "cosmiccleaned_" + STAMPS_FILENAME)
    new_stamps = data_stamps[~cosmics_mask, :]
    write_to_h5_file(h5_file, new_stamps)

    return

def _check_cosmics(inrow, xmatch_dist=4):
    """
    Internal function to detect cosmic rays in images - see clean_cosmics for full documentation

    :param inrow: slice of Pandas dataframe generated with .iterrows() method
    :param xmatch_dist: threshold at which to associate sources - in arcsec
    """

    idx, test = inrow
    print("starting #{}".format(idx)) # to verify program hasn't stalled

    try:
        head = fits.getheader(test.fits_filepath, "DIFFERENCE")
    except FileNotFoundError:
        try:
            head = fits.getheader(swap_dataloc(test.fits_filepath), "DIFFERENCE")
        except FileNotFoundError:
            frame_type = 99
            return idx, frame_type
    try:
        ncoadd = head["ncoadd"]
    except KeyError:
        frame_type = 99
        return idx, frame_type

    nearest_src = []

    # Loop over coadded frames
    for i in range(1, ncoadd + 1):
        coadd_file = head[f'coadd{i}']
        try:
            tempfits = fits.open(os.path.join(os.path.dirname(test.fits_filepath), coadd_file))
        except FileNotFoundError:
            try:
                tempfits = fits.open(os.path.join(os.path.dirname(swap_dataloc(test.fits_filepath)), coadd_file))
            except FileNotFoundError:
                frame_type = 99
                return idx, frame_type

        phottable = Table(tempfits["PHOTOMETRY"].data)
        testcoo = SkyCoord(phottable["ra"], phottable["dec"], unit=u.deg)
        srccoo = SkyCoord(test.ra, test.dec, unit=u.deg)
        _, d2d, _ = srccoo.match_to_catalog_sky(testcoo)  # by def this returns 1 object
        nearest_src.append(d2d.to(u.arcsec).value)

    # Return a bool vector len(ncoadds)
    has_src = np.squeeze(nearest_src) < xmatch_dist
    #has_src = [x < xmatch_dist for x in np.squeeze(nearest_src)]
    detnum = np.sum(has_src)

    if detnum == 1:
        frame_type = 0

    else:
        frame_type = 1

    return idx, frame_type

def clean_asteroids(data_dir, xmatch_dist, nproc):
    """
    Cross-match using SkyBoT to remove any asteroids that appear in the junk of a given dataset
    Use a large x-match distance to maximise purity.

    :param data_dir: directory containing the datastamps and labels
    :param xmatch_dist: threshold at which to associate sources - in arcsec
    :param nproc: number of processes to use
    :return:
    """

    data_stamps_path = os.path.join(data_dir, STAMPS_FILENAME)
    data_labels_path = os.path.join(data_dir, LABELS_FILENAME)

    # load table of labels
    data_labels = pd.read_csv(data_labels_path)

    junk_subset = data_labels[data_labels["label"] == 0]
    unique_imageids = np.unique(junk_subset.image_id)
    print("Querying {} unique images".format(len(unique_imageids)))

    with Pool(nproc) as pool:
        collated_dets = pool.starmap(_check_indiv_image, zip(unique_imageids, repeat(junk_subset), repeat(xmatch_dist)))

    clean_dets = [x for x in collated_dets if x is not None]
    check_tab = np.vstack(clean_dets)
    mp_ids = (check_tab[:, 0])[np.argwhere(check_tab[:, 1] == "True").squeeze()]

    good_mask = ~np.isin(data_labels.id, mp_ids)

    print("{} asteroids found in dataset".format(np.sum(~good_mask)))
    print("Writing new stamps and labels")

    new_table = data_labels.iloc[good_mask]
    new_table.to_csv(os.path.join(data_dir, "mpcleaned_" + LABELS_FILENAME), index=False)

    data_stamps = get_data_stamps(data_stamps_path)
    new_stamps = data_stamps[good_mask, :]

    write_to_h5_file(os.path.join(data_dir, "mpcleaned_" + STAMPS_FILENAME), new_stamps)
    return

def _check_indiv_image(iid, junk_subset, matchrad):
    """
    Internal function to check a given unique image id in a table for minor planets

    :param iid: image_id to check
    :param junk_subset: junk
    :param matchrad:
    :return:
    """
    table_subset = junk_subset[junk_subset.image_id == iid]
    test_item = table_subset.iloc[0]
    try:
        jd = fits.getheader(test_item.fits_filepath, 1)["jd"]
    except FileNotFoundError:
        jd = fits.getheader(swap_dataloc(test_item.fits_filepath), 1)["jd"]

    mp_res = query_skybot(test_item.ra, test_item.dec, 1.8, jd)
    if mp_res is None:
        print("Null table returned!")
        return None

    cand_coo = SkyCoord(table_subset.ra.values, table_subset.dec.values, unit=u.deg)
    mpc_ras = list(mp_res['RA(h)'])
    mpc_decs = list(mp_res['DE(deg)'])
    mpc_coo = SkyCoord(mpc_ras, mpc_decs, unit=(u.hourangle, u.deg))
    idx, d2d, d3d = cand_coo.match_to_catalog_sky(mpc_coo)
    match_mask = d2d.value < matchrad / 3600
    out = [(x["id"], res) for (_, x), res in zip(table_subset.iterrows(), match_mask)]
    print("Checked {} detections, {} asteroids found".format(len(table_subset), np.sum(match_mask)))
    return out

def remove_flagged_images(data_dir, flaglist, nkeep=10):
    """
    Remove flagged images from a dataset - optionally keep a set number from each image

    :param data_dir: directory where data stamps and labels are located
    :param flaglist: list of bad image ids, dumped from the marshall
    :param nkeep: number of detections to keep per bad frame
    :return:
    """

    data_labels = get_data_labels(os.path.join(data_dir, LABELS_FILENAME))
    bad_imgids = pd.read_csv(flaglist).values.flatten()
    cand_imgids = data_labels["image_id"].values

    print("Bad images in set: {}".format(
        np.sum(np.isin(cand_imgids, bad_imgids) & (data_labels["metalabel"] == "marshall"))))
    indiv_image_counts = [len(np.argwhere(cand_imgids == c)) for c in tqdm.tqdm(np.unique(cand_imgids))]
    print("Worst image - {} candidates in {}".format(np.max(indiv_image_counts),
                                                    np.unique(cand_imgids).astype(int)[np.argmax(indiv_image_counts)]))
    # for every bad image id
    bad_idxs = []

    for iid in tqdm.tqdm(bad_imgids):
        # if bad image and a marshall detection
        rel_idxs = np.argwhere((cand_imgids == iid) & (data_labels.metalabel.values == "marshall")).flatten()

        if len(rel_idxs) <= nkeep:
            continue

        bin_idxs = np.random.choice(rel_idxs, size=len(rel_idxs) - nkeep, replace=False)
        # print("Binned {} of {}".format(len(bin_idxs), len(rel_idxs)))
        bad_idxs += list(bin_idxs)

    good_mask = ~np.isin(np.arange(0, len(data_labels)), bad_idxs)
    testtab = data_labels.iloc[good_mask]

    print("Remaining Marshall detections: {}".format((testtab.metalabel.values == "marshall").sum()))
    print("Writing new labels")
    testtab.to_csv(os.path.join(data_dir, "clean_" + LABELS_FILENAME), index=False)
    print("loading old stamps")
    data_stamps = get_data_stamps(os.path.join(data_dir, STAMPS_FILENAME))
    new_stamps = data_stamps[good_mask, :]
    data_stamps_outloc = os.path.join(data_dir, "clean_" + STAMPS_FILENAME)
    print("writing new stamps")
    write_to_h5_file(data_stamps_outloc, new_stamps)
    return