import logging
import os
import sys
from itertools import repeat
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from gotorb import label_data
from gotorb.classifier import dropout_pred
from gotorb.utils import rb_colourmap

LOGGING_FORMAT = '%(asctime)s  %(levelname)-10s %(processName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class ClassifierImageVisual:
    def __init__(self, model_file, image, size, layers=None):
        """

        :param model_file: *.h5 model file of classifier output by keras
        :param image: filepath image to process
        :param size: stamp size of model
        :param layers: list of layers used from default 6 layers in `gotorb.label_data._get_stamps`
        """
        from tensorflow import keras

        self.model_file = model_file
        self.model = keras.models.load_model(model_file)
        self.image = os.path.abspath(image)
        self.rbthresh = 0.5
        self.posteriors = None

        self.basename = os.path.basename(image)
        self.dir_name = os.path.splitext(self.basename)[0]
        logger.info("initialising {}".format(self.basename))
        self.fits_file = fits.open(image)
        self.header = self.fits_file[1].header
        self.table = self.fits_file['photometry_diff'].data
        self.ndet = len(self.table)
        self.indiv_fits_files, self.indiv_images = self._get_indiv_fits_files()

        self.size = size
        if layers is None:
            layers = list(range(6))
        self.layers = layers

        self.norm_data_stamps = None
        self.indiv_data_stamps = None
        self.realbogus = None

    def _get_stamps(self, *args, **kwargs):
        return label_data._get_stamps(*args, **kwargs)

    def _get_indiv_fits_files(self):
        ncoadd = self.header.get('ncoadd')
        indiv_fits_files = []
        indiv_images = []
        if ncoadd is not None:
            for i in range(1, ncoadd+1):
                indiv_image = self.header['coadd{}'.format(i)]
                indiv_fits_files.append(fits.open(os.path.join(os.path.dirname(self.image), indiv_image)))
                indiv_images.append(indiv_image)
        else:
            indiv_fits_files.append(self.fits_file)
            indiv_images.append(self.image)

        return indiv_fits_files, indiv_images

    def main(self):
        logger.info("Running main for {}".format(self.basename))

        self.norm_data_stamps, self.indiv_data_stamps = self.make_data_stamps()
        self.realbogus = self.calcscores(self.norm_data_stamps)
        self.print_summary()
        self.make_jpgs()
        self.make_html()

    def marshall_ingest(self):
        logger.info("Running mock marshall ingestion for {}".format(self.basename))

        self.norm_data_stamps, self.indiv_data_stamps = self.make_data_stamps()
        self.realbogus = self.calcscores(self.norm_data_stamps)

        try:
            ingest_mask = (self.realbogus >= self.rbthresh) | (self.table["realbogus"] >= self.rbthresh) | \
                          (self.table["cnn_realbogus"] > self.rbthresh)
        except ValueError:
            ingest_mask = self.realbogus >= self.rbthresh

        self.print_summary()
        logger.info("{} sources with new RB >= {}".format(np.sum(self.realbogus >= self.rbthresh), self.rbthresh))
        logger.info("{} sources with rdf RB >= {}".format(np.sum(self.table["realbogus"] >= self.rbthresh), self.rbthresh))
        logger.info("{} sources with cnn RB >= {}".format(np.sum(self.table["cnn_realbogus"] > self.rbthresh), self.rbthresh))

        logger.info("{} of {} sources ingested".format(np.sum(ingest_mask), self.ndet))

        self.ndet = np.sum(ingest_mask)
        self.norm_data_stamps = self.norm_data_stamps[ingest_mask]
        self.indiv_data_stamps = self.indiv_data_stamps[ingest_mask]
        self.realbogus = self.realbogus[ingest_mask]
        self.table = self.table[ingest_mask]

        self.make_jpgs()
        # evaluate posteriors on those that passed the marshall cut
        self.posteriors = self.make_posteriors(self.norm_data_stamps)
        self.make_posterior_plots()

        self.make_html()

        print(Table(self.table))


    def make_data_stamps(self):
        from tensorflow import keras
        data_stamps = np.empty((self.ndet, self.size, self.size, 6))
        indiv_data_stamps = np.empty((self.ndet, self.size, self.size, len(self.indiv_fits_files)))
        logger.info("Making data stamps")

        # FIXME can't pickle fits.open object:
        # with Pool() as pool:
        #     data_stamps = pool.starmap(
        #         self._get_stamps,
        #         zip(
        #             repeat(self.fits_file),
        #             self.table['ra'],
        #             self.table['dec'],
        #             repeat(self.size),
        #             repeat(self.indiv_fits_files)
        #         ),
        #     )
        # data_stamps = np.array((data_stamps))

        for i, row in enumerate(self.table):
            print("{}/{}".format(i+1, self.ndet), end="\r")
            sys.stdout.flush()
            data_stamps[i, :, :, :] = self._get_stamps(self.fits_file, row['ra'], row['dec'],
                                                       self.size, self.indiv_fits_files)
            for j, indiv_fits_file in enumerate(self.indiv_fits_files):
                im_wcs = WCS(indiv_fits_file['IMAGE'].header)
                stamp_coo = SkyCoord(row['ra'], row['dec'], unit='deg')
                indiv_data_stamps[i, :, :, j] = Cutout2D(
                    indiv_fits_file['IMAGE'].data, stamp_coo, (self.size, self.size),
                    mode='partial', fill_value=0, wcs=im_wcs).data
        data_stamps = data_stamps[:, :, :, self.layers]
        logger.info("Normalising data stamps")
        norm_data_stamps = np.array(keras.backend.l2_normalize(data_stamps, axis=(1, 2)))

        return norm_data_stamps, indiv_data_stamps

    def calcscores(self, norm_data_stamps):
        logger.info("Make predictions")
        realbogus = self.model.predict(norm_data_stamps)[:, 0]

        return realbogus

    def make_posteriors(self, norm_data_stamps):
        logger.info("Using Monte Carlo dropout to generate posteriors")
        posteriors_new = dropout_pred(self.model, norm_data_stamps, nsamples=50)

        return posteriors_new

    def make_posterior_plots(self):
        logger.info("Generating posterior plots")

        for i, posterior in enumerate(self.posteriors):
            fig, ax = plt.subplots()
            ax.hist(posterior, bins='auto', density=True)
            ax.set_xlabel("realbogus")
            ax.set_ylabel("pdf")
            ax.axvline(0.5, c='k', ls='--')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, )
            ax.set_aspect(1 / ax.get_data_ratio())

            probstring = f"p_real : {np.sum(posterior > 0.5) / len(posterior):.3f}\np_bogus: {np.sum(posterior < 0.5) / len(posterior):.3f}"

            ax.annotate(probstring, xy=(0.05, 0.89), xycoords='axes fraction')
            outpath = os.path.join(self.dir_name, f"pos_{i}.png")
            fig.savefig(outpath, bbox_inches='tight')
            plt.close()


    def print_summary(self):
        logger.info("----------------- {} -----------------".format(self.basename))
        logger.info("Filename: {}".format(self.image))
        logger.info("Individual science images:")
        for indiv_image in self.indiv_images:
            logger.info("\t{}".format(indiv_image))
        logger.info("Difference image detections: {}".format(self.ndet))
        logger.info("Model: {}".format(self.model_file))
        logger.info("Realbogus distribution:")
        split = 50
        for i in range(10):
            upp = (i + 1) / 10.
            low = i / 10.
            n = np.sum((self.realbogus > low) * (self.realbogus <= upp))
            n_hash = split * n / self.ndet
            logger.info("{:.1f} < rb <= {:.1f} ({:5.0f}) {}".format(low, upp, n, "#"*int(n_hash)))
        logger.info("----------------{}-------------------".format("-"*len(self.basename)))

    def make_jpgs(self):
        logger.info("Making layer jpgs in {}".format(self.dir_name))
        with Pool() as pool:
            _ = pool.starmap(
                make_jpg,
                zip(
                    range(self.ndet),
                    self.norm_data_stamps,
                    repeat(self.dir_name)
                ),
            )
        logger.info("Making individual jpgs in {}".format(self.dir_name))
        with Pool() as pool:
            _ = pool.starmap(
                make_jpg,
                zip(
                    range(self.ndet),
                    self.indiv_data_stamps,
                    repeat(self.dir_name),
                    repeat(True),
                ),
            )

    def make_html(self):
        s = """
<html>
<body>
<style>
table {
border-spacing: 15px 2px;
}
tr:nth-child(even) {
  background-color: #f2f2f2
}
</style>
<p>Click sort by realbogus:</p>
<p><button onclick="sortTable()">Sort</button></p>
<table id="rbTable">
<tr>
"""
        s += "<th>Realbogus</th>"
        s += "<th>Posterior</th>"
        for j in range(len(self.layers)):
            s += "<th>Layer %s</th>" % j
        for k in range(len(self.indiv_fits_files)):
            s += "<th>Individual %s</th>" % k
        s += "</tr>"

        for i in range(self.ndet):
            s += "<tr>"

            rb = self.realbogus[i]
            rf = self.table["realbogus"][i]
            cnn = self.table["cnn_realbogus"][i]

            scores = [rb, rf, cnn]
            scorecols = [rb_colourmap(score) for score in scores]

            scorebox = ""
            for score, col in zip(scores, scorecols):
                scorebox += f"<span style='color:{col};'>{score:.3f}</span>\n"

            s += "<td style='font-size:2em; white-space:pre'>{}</td>".format(scorebox)

            # here's where the posterior will go
            s += f"<td> <img src='pos_{i}.png' width='{self.size*3}px'> </td>"

            for j in range(len(self.layers)):
                s += """<td>
<img src="%s_%s.jpg" width="%spx">
</td>
""" % (i, j, self.size*3)
            for k in range(len(self.indiv_fits_files)):
                s += """<td>
<img src="%s_indiv%s.jpg" width="%spx">
</td>
""" % (i, k, self.size*3)
            s += "</tr>"
        s += """
<script>
function sortTable() {
  var table, rows, switching, i, x, y, shouldSwitch;
  table = document.getElementById("rbTable");
  switching = true;
  /*Make a loop that will continue until
  no switching has been done:*/
  while (switching) {
    //start by saying: no switching is done:
    switching = false;
    rows = table.rows;
    /*Loop through all table rows (except the
    first, which contains table headers):*/
    for (i = 1; i < (rows.length - 1); i++) {
      //start by saying there should be no switching:
      shouldSwitch = false;
      /*Get the two elements you want to compare,
      one from current row and one from the next:*/
      x = rows[i].getElementsByTagName("TD")[0];
      y = rows[i + 1].getElementsByTagName("TD")[0];
      //check if the two rows should switch place:
      if (Number(x.innerHTML) > Number(y.innerHTML)) {
        //if so, mark as a switch and break the loop:
        shouldSwitch = true;
        break;
      }
    }
    if (shouldSwitch) {
      /*If a switch has been marked, make the switch
      and mark that a switch has been done:*/
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
    }
  }
}
</script>
</body>
</html>"""
        with open(os.path.join(self.dir_name, "index.html"), "w") as out:
            out.write(s)


def make_jpg(i, stamp, dir_name, indiv=False):
    os.makedirs(dir_name, exist_ok=True)
    for j in range(stamp.shape[2]):
        data = stamp[:, :, j]
        idata = scale_array_for_jpg(data)
        img = Image.fromarray(idata)
        img.convert(mode="RGB")
        draw = ImageDraw.Draw(img)
        size = img.size[0]
        # Draw central crosshairs
        d = 0.10  # cross hair line length as fractional size of image
        r = 0.07  # cross hair central hole radius as fractional size of image
        c = size / 2
        draw.line((c, c + size * r, c, c + size * (d + r)), fill='lightgreen', width=1)
        draw.line((c, c - size * r, c, c - size * (d + r)), fill='lightgreen', width=1)
        draw.line((c + size * r, c, c + size * (d + r), c), fill='lightgreen', width=1)
        draw.line((c - size * r, c, c - size * (d + r), c), fill='lightgreen', width=1)
        if indiv:
            filename = "{}_indiv{}.jpg".format(i, j)
        else:
            filename = "{}_{}.jpg".format(i, j)
        img.save(os.path.join(dir_name, filename))


def scale_array_for_jpg(array):
    zscale = ZScaleInterval()
    try:
        low, upp = zscale.get_limits(array)
    except IndexError:
        low, upp = np.nanpercentile(array, (1, 99))
    scaled_array = np.clip(array, low, upp)
    mi, ma = np.nanmin(scaled_array), np.nanmax(scaled_array)
    return ((scaled_array - mi) / (ma - mi) * ((2 << 7) - 1)).astype(np.uint8)
