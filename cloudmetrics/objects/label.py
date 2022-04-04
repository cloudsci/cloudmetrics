import skimage.measure as skmeasure


def label(mask, connectivity=1):
    object_labels = skmeasure.label(mask, connectivity=connectivity)
    return object_labels
