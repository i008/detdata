import os
import pathlib

import pandas as pd

from detdata.mxio import json_labels_to_csv
from .conftest import base_path, tr_csv, val_csv, template_path, testdata


def test_coco_json_labels_to_csv():
    json_labels_to_csv(testdata, output_csv_file=template_path,
                       val_split=0.0, shuffle=False)
    df = pd.read_csv(os.path.join(base_path, 'test_x_train.csv'))

    all_images = list(pathlib.Path(testdata).glob('*.jpg'))

    fns = [a.name for a in all_images]

    assert set(df.fname.tolist()) == set(fns)

    os.remove(tr_csv)
    os.remove(val_csv)
