=======
detdata
=======


.. image:: https://img.shields.io/pypi/v/detdata.svg
        :target: https://pypi.python.org/pypi/detdata

.. image:: https://img.shields.io/travis/i008/detdata.svg
        :target: https://travis-ci.org/i008/detdata

.. image:: https://readthedocs.org/projects/detdata/badge/?version=latest
        :target: https://detdata.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Tool to work with object detection data


* Free software: MIT license
* Documentation: https://detdata.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage




#### Create an index-csv for a coco-like json object detection dataset and write it to mxnet-records

get help:
```bash
python detdata/cli.py --help

```

example usage:

```bash
python detdata/cli.py --action parse_coco --coco-labels-dir /home/i008/googledrive/Projects/AiScope/malaria_dataset --out-path /home/i008/data

```
This will result in the following files beeing created:

```bash
tree /home/i008/data/
├── dataset_train.csv
├── dataset_train.mxindex
├── dataset_train.mxrecords
├── dataset_valid.csv
├── dataset_valid.mxindex
└── dataset_valid.mxrecords

```



