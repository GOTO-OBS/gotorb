gotorb
==============

create labelled data sets and train models to distinguish real/bogus detections in difference images

Installation and setup
----------------------

* From within `gotorb/` main directory:
  ```
  pip install --upgrade pip setuptools wheel
  pip install .
  ```

* Create a `.env` file in this directory following the example file `.env.example`
  (Requires a valid entry in your `~/.pgpass` file to connect to the `gotophoto` database 
  using these parameters.)
  
Pre-built validation datasets
---------------------
To enable easy testing of the `gotorb` code without access to internal GOTO data, we bundle a pre-built dataset.
This is roughly 10% of the classifier test set, bundled as a `hickle` file for ease of use.
This file can be downloaded from 
[https://files.warwick.ac.uk/tkillestein/browse/gotorb_validation_data
](https://files.warwick.ac.uk/tkillestein/browse/gotorb_validation_data).

To extract the data components from the `hickle`, use the below command.
```
stamps, meta = hickle.load("./datapack.hkl")
```  
The sample model can be loaded with the usual `tf.keras.models.load_model()` function.
A testing notebook to reproduce some of the figures in the accompanying paper is available in `notebooks/` -- the sample
data and model should be copied to the `notebooks` folder for use.
  
Creating labelled dataset
-------------------------
Our labelled dataset consists of a `csv` file with labels and some meta information on the detection, as well as
a `hdf5` file containing an array of image stamps for each detection in the `csv` file. 

##### Adapting the code to your own data source:
The key elements that need to be adjusted are `label_data.get_images()` and `label_data.get_individual_images()`
functions, and the `*_IMAGE_EXT` variables that set which FITS HDUs to look at. Some minor tweaks may also be needed
to the dataframe columns referenced depending on the way your data is laid out. More information about the functions
above can be found in the docstrings.

##### Generating own datasets (with gotoDB/fileserver access):
This must be done from a 
machine which can see the processed fits files of `gotophoto` (i.e. can see `/export/gotodata2/` etc.)

If we are using the offline minor planet checking by `pympc` we first need to download the catalogue so we can 
cross match to it:
```
import pympc
pympc.update_catalogue()
```

However, the above is not required for the default online checker, which uses 
[SkyBoT](http://vo.imcce.fr/webservices/skybot/).  
**Note**: do not use excessively, particularly with high thread counts (>10), as you will be throttled, 
and eventually blocked!

Then make our labelled data:
```
from gotorb import label_data
label_data.make(n_images=100)
```

See docstring of `label_data.make()` for parameters. This will produce a directory in the specified `out_dir` named
`data_YYYYMMDD-HHMMSS/` and containing:

 * `data_YYYYMMDD-HHMMSS.log` - a log file from the run
 * `data_labels.csv` - a csv file containing information on the detections as well as the assigned `label`
 * `data_stamps.h5` - a `hdf5` file containing the image stamps around each detection. 6 layers for each detection
   currently. 

*Note: If you get `OperationalError: fe_sendauth: no password supplied` when running `label_data.make()`,
check your database connection defined in `.env` has an authentication entry in `.pgpass` 
(see [here](https://www.postgresql.org/docs/9.3/libpq-pgpass.html).)*

The minor-planet labelled dataset can be supplemented using a csv file of photometry ids and labels (e.g. scraped
from the Marshall) with the `add_marshall_cands()` function.

Additionally, `label_data` can be used to create a training set of simulated transients and accompanying galaxy residuals
, using 
the `add_synthetic_transients()` function. These augmented stamps are produced by combining MP matches with nearby
galaxies in the image. Note that this requires a local copy of the [GLADE catalogue](http://glade.elte.hu/Download.html) 
in the catsHTM format, which must be specified in your `.env` file with the `CATSHTM_DIR` variable.

Train a model
-------------

Using our `data_YYYYMMDD-HHMMSS/` directory created by `label_data.make()` we will then train a model on this dataset.

```
from gotorb import classifier
model, meta = classifier.train("data_YYYYMMDD-HHMMSS")
```

There are lots of parameters, which can be passed to various parts of the keras model and fitter, see docstring for
more info. This will create directory name `data_YYYYMMDD-HHMMSS/model_YYMMDD-HHMMSS`, subsequent `.train()` calls
will be placed alongside with the timestamp of when the function was called. Inside the `model_YYMMDD-HHMMSS`
directory will be:

* `model_YYMMDD-HHMMSS.log` - a log file from the run
* `meta.hkl` - a file containing output from how the training went and extra information
  on the model parameters
* `model.h5` - the trained model weights, saved in keras `.h5` format.

Evaluate a model
----------------

There is a function to provide some metric and plots to evaluate how a trained model performed.

```
classifier.evaluate(model, meta)  # or you can use filepaths to the files created by `.train()`
```

This will produce a series of plots inside the `model_YYMMDD-HHMMSS` directory from which to evaluate the model.

For convenience, there is also a `benchmark` function which computes the same metrics as `evaluate` but on an arbitrary 
dataset in the same format as the training sets used.

There is also a comprehensive set of benchmarking tools available in the `notebooks` directory, including routines for 
comparing model performance against real-world datasets cross-matched with the Transient Name Server. These are provided
in Jupyter notebook form for ease of visualisation.

Visualising results
------------------

A preliminary script for eyeballing data stamps, their labels and the score from the trained model (and the old 
Random Forest classifier) is available with the `eyeball` module. This script can also be used to label new and re-label
existing datasets.

The `visualise` module evaluates a given model on an image, and can be used to run the classifier on a given image set.
This generates a HTML summary page with score and posterior information, and brings in the RF and CNN scores from 
existing classifiers.
This provides a useful sanity check that the classifier is performing optimally, and allows the developer to spot any
problems prior to deployment.

New: Bayesian Neural Networks
----------------------
You can train Bayesian versions of the `vgg6` model, using `bayesian_vgg6`. Tune the dropout parameter on
multiple runs to get the right level of regularisation (or use hyperparameter tuning), then use `classifier.dropout_pred()` to generate samples from
the score posteriors. For compatibility, these can be reduced to normal RB scores by averaging along axis 1, i.e.
the samples!

New: Tune model hyperparameters
----------------------
As a final step to extract maximum performance from the classifier, you can use the `hyperparameter_tuning` module to
find the optimal hyperparameter configuration. This will take a long time, but substantial gains in performance can be achieved!
See the docstring of `run_model_tuning` for more information on how to proceed.