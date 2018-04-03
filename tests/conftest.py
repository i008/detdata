import glob
import os

import pandas as pd
import pytest

from detdata.dgen import DetGen
from detdata.mxio import csv_to_mxrecords, json_labels_to_csv

base_path = os.path.dirname(os.path.realpath(__file__))
tr_csv = os.path.join(base_path, 'test_x_train.csv')
val_csv = os.path.join(base_path, 'test_x_valid.csv')

mxrecord = os.path.join(base_path, 'test_x_train.mxrecords')
mxindex = os.path.join(base_path, 'test_x_train.mxindex')
testdata = os.path.join(base_path, 'testdata')

def cleanup():
    print('cleaning up')
    os.remove(tr_csv)
    os.remove(val_csv)
    os.remove(mxrecord)

    os.remove(mxindex)


@pytest.fixture()
def create_mxrecords():
    json_labels_to_csv(testdata, output_csv_file=os.path.join(base_path,'test_x_{}.csv'), val_split=0.0, shuffle=False)
    df = pd.read_csv(tr_csv)

    fns = [a.split(os.sep)[-1] for a in list(glob.glob(os.path.join(base_path, 'testdata/*.jpg')))]
    assert set(df.fname.tolist()) == set(fns)
    csv_to_mxrecords(tr_csv, testdata, output_path=base_path)
    yield "teardown after that"

    cleanup()


@pytest.fixture()
def detgen_instance():
    csv_file = os.path.join(base_path,'test_x_{}.csv')
    json_labels_to_csv(testdata, output_csv_file=csv_file, val_split=0.0, shuffle=False)
    df = pd.read_csv(tr_csv)

    fns = [a.split(os.sep)[-1] for a in list(glob.glob(os.path.join(base_path, 'testdata/*.jpg')))]

    assert set(df.fname.tolist()) == set(fns)
    df_train = csv_to_mxrecords(tr_csv, testdata, output_path=base_path)

    g = DetGen(
        mxrecord,
        tr_csv,
        mxindex,
        batch_size=8
    )

    yield g

    cleanup()




