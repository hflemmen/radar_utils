import copy
import os
import pickle
import cv2
import numpy as np
import polarTransform
import scipy
from radar_utils.scan import PolarScan, EuclideanScan


def polarToEuc(scan: PolarScan, new_size=None, use_cache=False):
    img = None
    filepath = '/'.join(scan.img_path.split('/')[:-1])
    filename = scan.img_path.split('/')[-1].split('.')[0] + '.pkl'
    cart_path = f"{filepath}/radar_interference_filtered_cartesian/"
    cart_path_iter = cart_path + filename
    newScan = None
    if use_cache and os.path.exists(cart_path + filename):
        with open(cart_path_iter, 'rb') as f:
            newScan = pickle.load(f)
    else:
        new_img, settings = polarTransform.convertToCartesianImage(scan.img, imageSize=new_size,
                                                                   initialAngle=-np.pi / 2,
                                                                   finalAngle=3.0 / 2.0 * np.pi, hasColor=False,
                                                                   useMultiThreading=True)
        newScan = EuclideanScan(scan.sample_size,
                                scan.range_scales,
                                scan.range_scale_lengths,
                                False,
                                scan.timestamps[0],
                                scan.timestamps[-1],
                                scan.pose,
                                settings.center,
                                settings.finalRadius,
                                new_img.astype(np.uint8),
                                scan.clockwise)
    return newScan


def fillInRanges(scan: PolarScan) -> PolarScan:
    if scan.range_scales[0] != 1:
        raise ValueError("This function is not implemented for scans which are subsampled at the closest range.")
    new_img = scan.img[:, :scan.range_scale_lengths[0]]
    for i in range(1, len(scan.range_scales)):
        pixels_so_far = sum(scan.range_scale_lengths[:i])
        if pixels_so_far >= scan.img.shape[1]:
            break
        next_pixels = pixels_so_far + scan.range_scale_lengths[i]
        if next_pixels > scan.img.shape[1]:
            next_pixels = scan.img.shape[1]
        un_interpolated_img = scan.img[:, pixels_so_far:next_pixels]
        valuesToGetByInterpolation = np.array(range(0, un_interpolated_img.shape[1] * scan.range_scales[i])) / float(
            scan.range_scales[i])
        interpolated_segments = []
        for y in range(new_img.shape[0]):
            interpolated_segment = np.interp(valuesToGetByInterpolation, np.array(range(un_interpolated_img.shape[1])),
                                             un_interpolated_img[y, :])
            interpolated_segments.append(interpolated_segment)
        interpolated_img = np.array(interpolated_segments)
        new_img = np.hstack((new_img, interpolated_img))
    new_scan = scan.copyMeta()
    new_scan.img = new_img.astype(np.uint8)
    new_scan.range_scales = [1]
    new_scan.range_scale_lengths = [sum(scan.range_scale_lengths), ]
    return new_scan


def subsampleRange(scan: PolarScan, subsample_factor: int) -> PolarScan:
    if scan.range_scales[0] != 1:
        raise ValueError("This function is not implemented for scans which are subsampled at the closest range.")

    new_img = scan.img[:, :scan.range_scale_lengths[0]:subsample_factor]
    for i in range(1, len(scan.range_scales)):
        if scan.range_scales[i] > subsample_factor:
            break
        pixels_so_far = sum(scan.range_scale_lengths[:i])
        if pixels_so_far >= scan.img.shape[1]:
            break
        next_pixels = pixels_so_far + scan.range_scale_lengths[i]
        if next_pixels > scan.img.shape[1]:
            next_pixels = scan.img.shape[1]
        original_img = scan.img[:, pixels_so_far:next_pixels]
        if scan.range_scales[i] == subsample_factor:
            new_img = np.hstack((new_img, original_img))
            break
        else:
            subsampled_img = original_img[:, ::(subsample_factor / scan.range_scales[i])]
            new_img = np.hstack((new_img, subsampled_img))

    new_scan = scan.copyMeta()
    new_scan.img = new_img.astype(np.uint8)  # Possibly unnecessary to cast to uint8
    new_scan.range_scales = [1]
    new_scan.range_scale_lengths = [sum(scan.range_scale_lengths), ]
    new_scan.sample_size = scan.sample_size * subsample_factor
    return new_scan


if __name__ == '__main__':
    import radar_utils.viz as viz

    scan = PolarScan.load("/home/henrik/Data/polarlys/2018-06-15-17_41_30/Radar0/2018-06-15-17_41_30.bmp")

    new_scan = fillInRanges(scan)

    viz.disp_pol(new_scan)
