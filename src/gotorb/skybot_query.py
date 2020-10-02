# short script to query skybot for asteroid cross-matches

import warnings
from io import BytesIO
from time import sleep

import numpy as np
import requests
from astropy.io import ascii
from astropy.io.votable import parse_single_table
from astropy.table import Table


def query_skybot(ra, dec, rad, epoch, maglim=20.2, loc=950):
    """
    Query skybot, using text-parser instead of XML parser to avoid bugs.
    NB: be kind using this, as continuous querying with high thread counts will lead to **severe** throttling

    :param ra: RA in decimal degrees
    :param dec: DEC in decimal degrees
    :param rad: Search radius in decimal degrees
    :param epoch: Date of observation in JD
    :param maglim: Faintest objects we want to consider
    :param loc: MPC code for observatory
    :return: Table of detections, or None
    """

    querystring = "-ep={}&-ra={}&-dec={}&-loc={}&-rd={}&-output=object&-mime=text".format(epoch, ra, dec, loc, rad)
    queryurl = "http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?" + querystring

    # Delay request times to minimise maximum concurrent burst
    sleep(np.random.uniform(0, 1))

    httpreq = requests.get(queryurl)
    response = httpreq.text.split("\n")

    # Catch empty requests
    if len(response) <= 4:
        return None

    # Extract header and message content
    response_header = response[2].strip('#').split(' | ')
    response_table = response[3:]

    results_table = ascii.read(response_table, delimiter='|', names=response_header)
    results_table = Table(results_table)

    results_table = results_table[results_table["Mv"] < maglim]

    # Catch case if we have zero detections above mlim
    if len(results_table) == 0:
        return None

    return results_table


def skybot_request_legacy(ra, dec, rad, epoch, maglim=20.2, loc=950):
    """
    HTTP query to SkyBoT online checker
    :ra: RA in decimal degrees
    :dec: DEC in decimal degrees
    :rad: query radius in arcsec
    :epoch: observation date in JD
    :rtype: VOtable of cross-matches
    """

    querystring = "RA={}&DEC={}&SR={}&EPOCH={}&LOC={}".format(ra, dec, rad, epoch, loc)
    queryurl = "http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?" + querystring

    vo_req = requests.get(queryurl)
    vo_stream = BytesIO(vo_req.content)

    # This really shouldn't be necessary, but there's no way to suppress warnings about incompliant VOtables from
    # within astropy itself. The errors thrown don't relate to columns we use to cross-match, anyway
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vo_table = parse_single_table(vo_stream, pedantic=False).to_table()
            vo_table = vo_table[vo_table["magV"] < maglim]
            return vo_table

        # Catch multiple failure modes again from VOtable handling.
        # IndexError no table, ValueError empty table
        except (IndexError, ValueError):
            return None


def query_skybot_legacy(ra, dec, rad, epoch, maglim=20.2, loc=950, maxreq=5):
    """
    Wrapper for querying skybot using the skybot_request handler above. Queries seem to randomly drop, so
    multiple tries are allowed before giving up.

    :param ra: RA in decimal degrees
    :param dec: DEC in decimal degrees
    :param rad: Search radius in decimal degrees
    :param epoch: Date of observation in JD
    :param maglim: Faintest objects we want to consider
    :param loc: MPC code for observatory
    :param maxreq: Maximum number of times to try to skybot_request
    :return: VOtable of detections, or None
    """
    for i in range(maxreq):
        tab = skybot_request_legacy(ra, dec, rad, epoch, maglim=maglim, loc=loc)
        if tab is not None:
            return tab
        sleep(0.05)  # Hard coded rate limit - adjust as required to avoid blacklisting.

    return None
