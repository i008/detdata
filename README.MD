#### Create an index-csv for a coco-like json object detection dataset and write it to mxnet-records
![[Build Status](https://travis-ci.org/i008/detdata.svg?branch=master)](https://travis-ci.org/i008/detdata)

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

