import numpy as np


__all__ = ['VOCToClassVector']


VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


class VOCToClassVector(object):
    """Transformation from VOC annotation dict to a binary class vector."""
    def __init__(self):
        self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
        self.num_classes = len(VOC_CLASSES)

    def __call__(self, d):
        # Verify annotation dict.
        assert 'annotation' in d
        assert 'object' in d['annotation']

        # Ensure objects are in a list.
        objs = d['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        # Prepare classification vector.
        class_vector = np.zeros(self.num_classes, dtype=np.float32)
        for obj in objs:
            assert 'name' in obj
            class_i = self.class_to_idx[obj['name']]
            class_vector[class_i] = 1.

        return class_vector
