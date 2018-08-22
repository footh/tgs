from skimage import transform
from skimage import filters


def downsample(arr, resize_method, resize_param):
    """
       Downsamples image to proper size based on method
    """
    if resize_method == 'pad' or resize_method == 'pad-fixed':
        top = resize_param[0, 0]
        bottom = resize_param[0, 1]
        left = resize_param[1, 0]
        right = resize_param[1, 1]
        r, c = arr.shape
        return arr[top:r - bottom, left:c - right]
    else:
        return transform.resize(arr, (resize_param, resize_param), mode='constant', preserve_range=True)


def threshold(preds, method=None, prob_thresh=0.5):
    if method == 'otsu':
        thresh = filters.threshold_otsu(preds)
        return preds > thresh
    else:
        return preds > prob_thresh
