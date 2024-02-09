import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='gotorb',
    version='1.0',
    author='Thomas Killestein',
    description='create labelled datasets and train models to distinguish real/bogus detections',
    packages=['gotorb'],
    package_dir={'gotorb': 'src/gotorb'},
    install_requires=[
        "tensorflow>=2.1.0",
        "scikit-learn~=0.21",
        "pandas>=0.25.1",
        "psycopg2-binary==2.8.3",
        "astropy==3.2.1",
        "hickle==3.4.5",
        "tables==3.6.1",
        "pympc==0.6.1",
        "python-decouple==3.1",
        "matplotlib==3.1.1",
        "keras-tuner~=1.0.0",
        "h5py==2.10.0"
    ],
    license='GNU General Public License v3 (GPLv3)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
         "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6,<3.8',
)
