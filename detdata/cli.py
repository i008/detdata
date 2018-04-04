import os
import begin
from detdata.mxio import csv_to_mxrecords, json_labels_to_csv


@begin.subcommand
def parse_coco_like(coco_labels_dir: 'dir to coco-like ds' = None, out_path: 'target path' = None):
    if coco_labels_dir is None or out_path is None:
        raise ValueError("Please provide neccesery input type  python cli.py parse_coco_like --help for help")

    csv_out = os.path.join(out_path, 'dataset_{}.csv')
    json_labels_to_csv(coco_labels_dir, output_csv_file=csv_out)

    csv_train = csv_out.format('train')
    csv_valid = csv_out.format('valid')

    csv_to_mxrecords(csv_train, coco_labels_dir, out_path)
    csv_to_mxrecords(csv_valid, coco_labels_dir, out_path)


@begin.subcommand
def csv_to_mxindex(csv_index_file: 'CSV index file with annotations',
                   base_dir: 'Directory where the images mentioned in csv index are',
                   output_path: 'Where to save mxindex and mxrecord files'):
    """

    :param csv_index_file: Csv file with annotations and filenames
    :param base_dir: base_dir joined  with fname  in csv should give  a valid path to the image
    :param output_path: where to store mxrecord files
    :return:
    """

    csv_to_mxrecords(csv_index_file, base_dir, output_path)


@begin.start
def main():
    pass
