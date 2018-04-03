import os
import begin
from mxio import csv_to_mxrecords, json_labels_to_csv


@begin.start
def main(action: 'Action: one of [parse_coco_like,]' = 'parse_coco_like',
         coco_labels_dir: 'dir to coco-like ds' = '.',
         out_path: 'target path' = ''):

    """Run parsing json labels"""

    if action == 'parse_coco_like':
        csv_out = os.path.join(out_path, 'dataset_{}.csv')
        json_labels_to_csv(coco_labels_dir, output_csv_file=csv_out)

        csv_train = csv_out.format('train')
        csv_valid = csv_out.format('valid')

        csv_to_mxrecords(csv_train, coco_labels_dir, out_path)
        csv_to_mxrecords(csv_valid, coco_labels_dir, out_path)
