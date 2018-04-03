import mxnet
import numpy as np
from numpy.testing import assert_almost_equal

from .conftest import mxindex, mxrecord


def test_generator(create_mxrecords):
    reader = mxnet.recordio.MXIndexedRecordIO(mxindex, mxrecord, 'r')
    H, I = mxnet.recordio.unpack_img(reader.read())
    print(create_mxrecords)
    print(H.label.shape)
    assert I.shape == (750, 750, 3)


def test_csv_index_corresponds_with_mxrecord_index(detgen_instance):
    """
    Checks that the index of mxrecord files is the same as in the csv-file.
    """

    image_id = 3
    row = detgen_instance.index_df.loc[image_id].iloc[0]
    a = np.array([row.xmin, row.ymin, row.xmax, row.ymax])

    h, im = detgen_instance.read_one(3)
    assert_almost_equal(a, h[0][1:])


def test_augmentation_resize(create_mxrecords):
    pass


def test_retina_net_comp_generator(detgen_instance):

    retina_gen = detgen_instance.get_retina_comp_generator()
    ims, label = next(retina_gen)
    assert label[0][0][-1] == 0
    assert len(label[0][0]) == 5
