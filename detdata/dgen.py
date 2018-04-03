import logging

import imgaug as ia
import mxnet as mx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class DetGen:
    """

    """

    def __init__(
            self,
            mx_record, csv_index, mx_index,
            batch_size=1,
            to_bgr=False,
            cutoutbbs=True,
            removebbs=False):
        """
        :param mx_record:
        :param csv_index:
        :param mx_index:
        :param batch_size:
        """
        self.mx_record = mx_record
        self.csv_index = csv_index
        self.mx_index = mx_index
        self.batch_size = batch_size
        self.to_bgr = to_bgr
        self.cutoutbbs = cutoutbbs
        self.removebbs = removebbs

        self.reader = mx.recordio.MXIndexedRecordIO(
            idx_path=self.mx_index,
            uri=self.mx_record,
            flag='r'
        )

        self.n_classes = 1
        self.max_number_of_boxes = 20
        self.index_df = pd.read_csv(csv_index).set_index('image_id')
        self.index_df.class_name = self.index_df.class_name.astype('category')
        self.index_df['class_id'] = self.index_df.class_name.cat.codes

        drops = self.index_df.drop_duplicates('class_name')
        self.classes = dict(zip(drops['class_name'], drops['class_id']))

    @staticmethod
    def raw_sequentail_generator(reader, bs, to_bgr=False):
        """

        :param reader:
        :param bs:
        :return:
        """
        images, boxes_collect = [], []

        while True:
            record = reader.read()
            if record is None:
                reader.reset()
            else:
                header, image = mx.recordio.unpack_img(record)
                if to_bgr:
                    image = image[:, :, ::-1]
                boxes = header.label[2:].reshape(-1, 5)
                images.append(image)
                boxes_collect.append(boxes)
            if len(images) == bs:
                yield images, boxes_collect
                images, boxes_collect = [], []

    @staticmethod
    def array_of_boxes_to_imgaug(array_of_boxes, size_threshold=200):
        """

        :param array_of_boxes:
        :param size_threshold:
        :return:
        """
        imgaug_boxes = []
        classes = []
        for box in array_of_boxes:
            class_id = box[0]
            try:
                imgaug_box = ia.BoundingBox(*box[1:])
                if imgaug_box.area < size_threshold:
                    # this is needed if you have buggy markings if bbs are to small augmentation might fail
                    continue
            except AssertionError:
                logger.deubug("found corrupted box {}".format(box[1:]))
                continue
            else:
                imgaug_boxes.append(imgaug_box)
                classes.append(class_id)
        return imgaug_boxes, classes

    @staticmethod
    def batch_of_boxes_to_imgaug_format(images, list_of_image_boxes):
        """

        :param images:
        :param list_of_image_boxes:
        :return:
        """
        shapes = [im.shape for im in images]

        img_aug_boxes = []
        classes_collect = []
        for image_boxes in list_of_image_boxes:
            imgaug_boxes, classes = DetGen.array_of_boxes_to_imgaug(image_boxes)
            img_aug_boxes.append(imgaug_boxes)
            classes_collect.append(classes)

        bounding_boxes_on_image = []
        for shape, boxes in zip(shapes, img_aug_boxes):
            bbs = ia.BoundingBoxesOnImage(boxes, shape=shape)
            bounding_boxes_on_image.append(bbs)
        return images, bounding_boxes_on_image, classes_collect

    @staticmethod
    def imgaug_bbs_labels_to_padded_arrays(list_bbs, list_classes, bs, max_n_boxes, scale_labels=256):
        """
        list_bbs = [[bounding_boxes_on_image,...]
        list_classes = [[c1..]c2..]]

        """
        mask = np.zeros((bs, max_n_boxes, 5)) + -1
        for batch_id, (bbs, classes) in enumerate(zip(list_bbs, list_classes)):
            for bb_id, (box, c) in enumerate(zip(bbs.bounding_boxes, classes)):
                b = np.array([box.x1, box.y1, box.x2, box.y2]) / scale_labels  # !
                mask[batch_id, bb_id] = np.append(c, b)

        return mask

    @staticmethod
    def mxnet_to_tf(mxnet_image_batch):
        """
        Images in mxnet are by default in BGR and channels first format. This changes it to channels last and RGB
        :param mxnet_image_batch:
        :return:
        """
        channels_last = mxnet_image_batch.transpose([0, 2, 3, 1])
        bgr_to_rgb = channels_last[:, :, :, ::-1].astype('uint8')
        return bgr_to_rgb

    @staticmethod
    def tf_to_mxnet(tf_image_batch):
        """

        :param tf_image_batch:
        :return:
        """
        channel_first = tf_image_batch.transpose([0, 3, 1, 2])
        rgb_to_bgr = channel_first[:, ::-1, :, :]

        return rgb_to_bgr

    @staticmethod
    def retina_net_comp_labels(list_of_imgaug_bb, list_of_bb_cls):
        collect_boxes_for_batch = []
        for batch_id, (bbs, classes) in enumerate(zip(list_of_imgaug_bb, list_of_bb_cls)):
            collect_boxes_for_image = []
            for bb_id, (box, c) in enumerate(zip(bbs.bounding_boxes, classes)):
                onebox = np.array([box.x1, box.y1, box.x2, box.y2, c])
                collect_boxes_for_image.append(onebox)
            collect_boxes_for_batch.append(np.vstack(collect_boxes_for_image))
        return collect_boxes_for_batch

    def get_raw_generator(self):
        """

        :return:
        """
        return self.raw_sequentail_generator(self.reader, self.batch_size, self.to_bgr)

    def get_augmenting_generator(self, augmenter=None):
        """

        :param augmenter:
        :return:
        """
        g = self.get_raw_generator()
        while True:
            im, boxes = next(g)
            im_imgaug, bbs_imgaug, classes = self.batch_of_boxes_to_imgaug_format(im, boxes)
            if augmenter:
                augmenter = augmenter.to_deterministic()
                ims_aug = augmenter.augment_images(im_imgaug)
                try:
                    bbs_aug = augmenter.augment_bounding_boxes(bbs_imgaug)
                    if self.cutoutbbs:
                        bbs_aug = [b.cut_out_of_image() for b in bbs_aug]
                    if self.removebbs:
                        bbs_aug = [b.remove_out_of_image() for b in bbs_aug]
                except AssertionError:
                    logger.exception("""
                    augmenter generated corrupted boxees skipping batch usually happens bc of resizing v small boxes
                    """)
                    continue
                except Exception as e:
                    logger.exception("Unknown error", bbs_imgaug)
                    continue
                else:
                    yield ims_aug, bbs_aug, classes
            else:
                yield im_imgaug, bbs_imgaug, classes

    def get_mxnet_generator(self, augmenter=None, mxnet_context=mx.cpu()):
        """

        :param augmenter:
        :param mxnet_context:
        :return:
        """
        g = self.get_augmenting_generator(augmenter=augmenter)
        while True:
            im, boxes, classes = next(g)
            im = np.stack(im, axis=0)
            input_images = mx.nd.array(DetGen.tf_to_mxnet(im), ctx=mxnet_context)
            output_bbs = DetGen.imgaug_bbs_labels_to_padded_arrays(
                boxes,
                classes,
                self.batch_size,
                self.max_number_of_boxes)
            output_bbs = mx.nd.array(output_bbs, ctx=mxnet_context)
            yield input_images, output_bbs

    def get_retina_comp_generator(self, augmenter=None):
        g = self.get_augmenting_generator(augmenter=augmenter)
        while True:
            images, boxes, classes = next(g)
            annotations = self.retina_net_comp_labels(boxes, classes)

            yield images, annotations

    def read_one(self, idx):
        header, image = mx.recordio.unpack_img(self.reader.read_idx(idx))
        header = header.label[2:].reshape((-1, 5))
        return header, image
