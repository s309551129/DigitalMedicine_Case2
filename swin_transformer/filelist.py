import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset

classes = [
    '0',
    '1',
    '2'
]
classes_to_idx = {classes[i]: i for i in range(len(classes))}

@DATASETS.register_module()
class Filelist(BaseDataset):
    classes = [
        '0',
        '1',
        '2'
    ]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                label = classes_to_idx[gt_label]
                info['gt_label'] = np.array(label, dtype=np.int64)
                data_infos.append(info)
            return data_infos