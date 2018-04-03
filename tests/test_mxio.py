from detdata.mxio import json_labels_to_csv, csv_to_mxrecords
import pandas as pd
import glob
import os
import pytest
import mxnet
from .conftest import base_path, tr_csv, val_csv


def test_coco_json_labels_to_csv():
    json_labels_to_csv(os.path.join(base_path, 'testdata'), output_csv_file=os.path.join(base_path, 'test_x_{}.csv'),
                       val_split=0.0, shuffle=False)
    df = pd.read_csv(os.path.join(base_path, 'test_x_train.csv'))

    fns = [a.split(os.sep)[-1] for a in list(glob.glob(os.path.join(base_path, 'testdata/*.jpg')))]

    assert set(df.fname.tolist()) == set(fns)

    os.remove(tr_csv)
    os.remove(val_csv)
