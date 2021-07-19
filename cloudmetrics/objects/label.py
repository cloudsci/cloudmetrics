import skimage.measure as skmeasure


def label(cloud_mask, connectivity=1):
    cloud_object_labels = skmeasure.label(cloud_mask, connectivity=connectivity)
    return cloud_object_labels
