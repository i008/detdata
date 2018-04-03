import json
import logging
import os
import pathlib
from itertools import chain

from PIL import Image
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet.recordio import MXIndexedRecordIO
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

RANDOM_STATE = 666
import pdb

def json_labels_to_csv(
        dataset_path,
        output_csv_file='dataset_{}.csv',
        output_format=['class_name', 'fname', 'xmin', 'ymin',
                       'xmax', 'ymax','im_rows','im_cols','image_id'],
        val_split=0.1,
        shuffle=True
):
    """

    :param dataset_path: Path to coco-like dataset (each pic has a json label file)
    :param output_csv_file:
    :param output_format: order of the columns in the csv-file
    defaults to: ['class_name', 'fname', 'xmin', 'ymin', 'xmax', 'ymax']
    :param val_split: frac=0..1 size of validation data split
    :param shuffle:
    :return:
    """
    dataset_path = pathlib.Path(dataset_path)
    images = list(dataset_path.glob('*.jpg'))
    records = []
    for i, image_path in tqdm(enumerate(images)):
        file_name = image_path.name
        image = dataset_path / (image_path.stem + '.jpg')
        im = Image.open(image)
        im_rows, im_cols = im.size
        annotation = dataset_path / (image_path.stem + '.json')
        annotation_dict = json.loads(open(str(annotation)).read())
        boxes = annotation_dict['boxes']
        for box in boxes.keys():
            current_box = boxes[box]
            record = {
                'fname': file_name,
                'class_name': current_box['illness'],
                'xmin': int(current_box['xmin']),
                'ymin': int(current_box['ymin']),
                'xmax': int(current_box['xmax']),
                'ymax': int(current_box['ymax']),
                'im_rows': im_rows,
                'im_cols': im_cols,
                'image_id': i
            }

            records.append(record)

    df = pd.DataFrame(records)
    df = df[output_format]

    # since the dataset contains one box per row to achieve a fair split we split
    # based on file names
    valid_names = df.drop_duplicates('fname').fname.sample(frac=val_split, random_state=RANDOM_STATE)

    df_valid = df[df.fname.isin(valid_names)]
    df_train = df[~(df.fname.isin(valid_names))]

    # in case we loaded the data  in a particular order, lets shuffle the index.
    # as we will write it sequantially to mxrecord
    if shuffle:
        df_valid = df_valid.sample(frac=1, random_state=RANDOM_STATE)
        df_train = df_train.sample(frac=1, random_state=RANDOM_STATE)

    assert df_train.shape[0] + df_valid.shape[0] == df.shape[0]

    logger.info("Writing train index csv".format(output_csv_file))
    df_train.to_csv(output_csv_file.format('train'), header=True, index=False)
    logger.info("Writing valid index csv".format(output_csv_file))
    df_valid.to_csv(output_csv_file.format('valid'), header=True, index=False)


def _parse_labels_csv_for_mxrecords_writers(labels_csv, base_path, recordio_comp=True):
    """
    Mxrercod writers need some special data preparation (for example prepending labels with the header size)

    :param labels_csv: csv index
    :param base_path: where the files_names (mentioned in labels_csv) actually are
    :param recordio_comp: if prepend [2, 5] (mxrecord compatibility)
    :return:
    """

    labels_df = pd.read_csv(labels_csv)
    labels_df.fname = labels_df.fname.apply(lambda file_name: os.path.join(base_path, file_name))
    labels_df.class_name = labels_df.class_name.astype('category')
    labels_df['class_id'] = labels_df.class_name.cat.codes
    labels_df['comb'] = labels_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    labels_df['comb'] = labels_df.class_id.apply(lambda x: [x]) + labels_df.comb

    # labels_df contains 1 bbox per row, to easier iterate over it later we combine all boxes
    # for a n image into one list [class_id, xmin, ymin, xmax ,ymax, class_id_i, xmin, ymin,  xmax,ymax]
    # len of comb_lists=(Nbbofx * 5)
    comb_df = pd.DataFrame(
        labels_df.groupby('fname')['comb'].apply(lambda x: list(chain(*x)))
    ).reset_index()

    comb_df['id'] = comb_df.index
    comb_df = pd.merge(comb_df, labels_df[['fname', 'image_id']], on='fname').drop_duplicates(subset='fname')

    # this is needed for mx.recordio - it expects the label vector to contain [header_length, cycle_length]
    # so for instance [2, 5] means  header is 2 digits long (len([2,5])) and each bounding box is described with
    # 5 numbers [class_id, xmin, ymin, xmax, ymax]
    if recordio_comp:
        comb_df['comb'] = [2, 5] + comb_df.comb

    return comb_df


def csv_to_mxrecords(labels_csv, base_path, output_path='.'):
    """
    Writes images and annotations(bboxes) into fast mxrecord files. From csv index.
    (Can be created with coco_json_labels_to_csv)

    csv schema: ['class_name[str]','fname[str]', 'xmin','ymin','xmax','ymax']

    :param labels_csv: csv file with ['class_name[str]','fname[str]', 'xmin','ymin','xmax','ymax']
    :param base_path: Path to where fnames resids in filesystem.
    :param output_path: where to write the mxrecord file
    :return:
    """
    stem = pathlib.Path(labels_csv).stem
    comb_df = _parse_labels_csv_for_mxrecords_writers(labels_csv, base_path)

    indedx_path = os.path.join(output_path, '{}.mxindex'.format(stem))
    records_path = os.path.join(output_path, '{}.mxrecords'.format(stem))
    writer = MXIndexedRecordIO(
        indedx_path,
        records_path,
        flag='w'
    )

    for i, row in tqdm(comb_df.iterrows(), total=comb_df.shape[0]):
        image = np.array(Image.open(row.fname))
        image_header = mx.recordio.IRHeader(flag=0, label=row.comb, id=row.image_id, id2=0)
        img_record = mx.recordio.pack_img(image_header, image, quality=100, img_fmt='.jpg')
        writer.write_idx(row.image_id, img_record)
    writer.close()

    return comb_df
