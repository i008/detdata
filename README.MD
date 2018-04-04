![alt text](https://api.travis-ci.org/i008/detdata.svg "")

#### Create an index-csv for a coco-like json object detection dataset and write it to mxnet-records

get help:
```bash
python detdata/cli.py --help

```

example usage:

```bash
python detdata/cli.py parse_coco --coco-labels-dir /home/i008/googledrive/Projects/AiScope/malaria_dataset --out-path /home/i008/data

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

#### Create mxrecords and mxindex file from a csv

```bash
python detdata/cli.py csv_to_mxindex

```


#### Basic usage - raw geneator

This will yield list of the length of the batch size specified.  With not processed images and
nd arrays of bounding boxes in the format ('class_id','xmin','ymin','xmax','ymax')

```python

detdata = DetGen(
    '/path/to/dataset.mxrecords',
    '/path/to/dataset_train.csv',
    '/path/to/dataset_train.mxindex',
    batch_size=8
)

raw_generator = detdata.get_raw_generator()

list_images, list_bboxes = next(raw_generator)
print('class_id--xmin--ymin--xmax--ymax')
print(list_bboxes[0])

class_id  xmin  ymin  xmax  ymax
[[   0.  190.  502.  230.  542.]
 [   0.   16.  261.   56.  301.]
 [   0.  221.  475.  261.  515.]
 [   0.  111.  619.  151.  659.]]

```


#### Basic usage - augmented generator

You can use imgaug (https://github.com/aleju/imgaug) as your augmentation engine
you can find its documentation here: http://imgaug.readthedocs.io/en/latest/

```python

import imgaug as ia
from imgaug import augmenters as iaa

# create a dummy seqiantial imgaug.augmenter
dummy_augmenter = iaa.Sequential([iaa.Noop()])
aug_generator = detdata.get_augmenting_generator(augmenter=dummy_augmenter)
images, bbs_on_image, classes  = next(aug_generator)
```

bbs_on_image contains a list of imgaug.imgaug.BoundingBoxesOnImage objects.





