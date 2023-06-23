from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np
import cv2
import os
import datetime


class ScanRep(Enum):
    POLAR = 1
    EUCLEDIAN = 2


@dataclass
class PolarScan:
    """
    Class to represent a single radar scan.
    """
    img_path: str
    sample_size: float  # The distance each pixel in the radar image corresponds to.
    spokes: int  # The number of radar spokes, i.e. the number of rows in the polar image
    range_scales: "List[int]"  # The multiplier to multiply with the sample size at this range
    range_scale_lengths: "List[int]"  # How many pixels to use this multiplier for
    timestamps: "List[int]"
    azimuths: "List[float]"  # List of all the directions of the rows of the polar image in radians
    pose: "Tuple[float, float, float]"  # The GNSS pose at the start of the scan on the format (lat [decimal], long [decimal], heading [0-2pi][radians],). 0 heading is north and pi/2 is west.
    img: np.ndarray
    clockwise: bool = False  # True if the radar is spinning clockwise and False if the radar is spinning counter clockwise

    # If it is clockwise, it also implies that the azimuth angles are in the opposite direction

    def copyMeta(self):
        return PolarScan(img_path=self.img_path, sample_size=self.sample_size, spokes=self.spokes,
                         range_scales=self.range_scales, range_scale_lengths=self.range_scale_lengths,
                         timestamps=self.timestamps, azimuths=self.azimuths, pose=self.pose, img=np.array([]),
                         clockwise=self.clockwise)

    @staticmethod
    def load(img_path, use_cache=False, clockwise=False):
        img = None
        filepath = '/'.join(img_path.split('/')[:-1])
        filename = img_path.split('/')[-1]
        interferece_path = f"{filepath}/radar_interference_filtered/"
        radar_interference_filter_done = False
        if use_cache:
            if os.path.exists(interferece_path + filename):
                img = cv2.imread(interferece_path + filename, cv2.IMREAD_GRAYSCALE)
                radar_interference_filter_done = True
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        path = img_path.split('.')[0]

        with open(path + ".txt", 'r') as f:
            f.readline()
            header = f.readline().split(' ')
            lines = f.readlines()

        timestamps = []
        azimuths = []
        range_scales = [1]
        range_scale_lengths = [img.shape[1]]
        for line in lines:
            line = line.split(',')
            if len(timestamps) == 0:
                pose = (float(line[3]), float(line[4]), float(line[2]),)
                sample_size = float(line[5].strip())  # We assume same sample size for the whole scan

            timestamps.append(datetime.datetime.fromisoformat(line[0]))
            azimuth_radians = float(line[1])
            azimuths.append(azimuth_radians)

        scan = PolarScan(img_path, sample_size, len(lines), range_scales, range_scale_lengths, timestamps, azimuths,
                         pose, img, clockwise)
        if radar_interference_filter_done or not use_cache:
            return scan
        else:
            raise NotImplementedError("Radar interference filter not cached")

    def unproject_coordinate(self, pt: np.ndarray):
        """
        Calculates the (x, y) [m, m] coordinate relative to the ships position from the pixel coordinate in the polar scan.
        x is forward and y is starboard.
        """
        range_px, angle_idx = tuple(pt)
        range_m = self.get_range(range_px)
        angle = self.get_angle(angle_idx)
        res = np.array([np.cos(angle) * range_m, np.sin(angle) * range_m, ])  # (x, y)
        return res

    def get_range(self, range_px: int):
        """
        Calculates the distance from the ship of a point given the pixel x coordinate in the polar image of the scan.
        """
        range_m = range_from_scaled_lengths(range_px, self.range_scales, self.range_scale_lengths, self.sample_size)
        return range_m

    def get_angle(self, angle_idx):
        """
        Gets the angle in radians of the y pixel coordinate in the polar image
        """
        # TODO: Interpolate between the azimuth directions.
        if self.clockwise:
            orientation = 1
        else:
            orientation = -1
        return orientation * self.azimuths[int(angle_idx % len(self.azimuths))]

    def get_range_angle(self, pt):
        """
        Gets the real world angle and range from the real world d
        """
        range_px, angle_idx = tuple(pt)
        return self.get_range(range_px), self.get_angle(angle_idx)


def linear_interpolation(img, pos):
    p0 = np.floor(pos).astype(np.int)
    alpha = pos - p0
    return img[tuple(p0 + np.array([0, 0]))] * (1 - alpha[0]) * (1 - alpha[1]) + \
        img[tuple(p0 + np.array([1, 0]))] * alpha[0] * (1 - alpha[1]) + \
        img[tuple(p0 + np.array([0, 1]))] * (1 - alpha[0]) * alpha[1] + \
        img[tuple(p0 + np.array([1, 1]))] * alpha[0] * alpha[1]


def range_from_scaled_lengths(range_px, range_scales, range_scale_lengths, sample_size):
    range_m = 0
    for i in range(len(range_scales)):
        pixels_so_far = sum(range_scale_lengths[:i])
        pixels_so_far_and_this = sum(range_scale_lengths[:i + 1])
        if range_px > pixels_so_far_and_this:
            range_m += range_scales[i] * range_scale_lengths[i] * sample_size
        else:
            range_m += (range_px - pixels_so_far) * range_scales[i] * sample_size
            return range_m
    raise ValueError("You gave in a range that is too large for this image.")


def px_coord_from_scaled_lenghts(range_m, range_scales, range_scale_lengths, sample_size):
    pixels_so_far = 0
    range_so_far = 0
    for i in range(len(range_scales)):
        range_so_far_and_this = range_so_far + range_scales[i] * range_scale_lengths[i] * sample_size

        if range_m > range_so_far_and_this:
            range_so_far = range_so_far_and_this
        else:
            pixels_so_far = sum(range_scale_lengths[:i])
            range_at_this_scale = range_m - range_so_far
            pixels_at_this_scale = range_at_this_scale / (range_scales[i] * sample_size)
            return pixels_so_far + pixels_at_this_scale

    raise ValueError("You gave in a range that is too large for this image.")


@dataclass
class EuclideanScan:
    """
    Class to represent a single radar scan.
    """
    sample_size: float  # (m) Each pixel is sample_size^2 lage
    range_scales: "List[int]"  # The multiplier to multiply with the sample size at this range
    range_scale_lengths: "List[int]"  # How many pixels to use this multiplier for
    true_size: bool  # Is the sample size correct
    timestamp_start_capture: int  # The timestamp of the oldest radar spoke in the scan
    timestamp_end_capture: int  # The timestamp of the youngest radar spoke in the scan
    pose: "Tuple[float]"  # The GNSS pose at the start of the scan on the format (lat [decimal], long [decimal], heading [0-2pi][radians],). 0 heading is north and pi/2 is west.
    center: "np.ndarray[2]"  # The pixel coordinate of the ship in the radar image
    radius: "np.ndarray[2]"  # The valid radius of the image. Every pixel more than "radius" away from "center" is invalid.
    img: np.ndarray  # The scan itself
    clockwise: bool = False  # True if the radar is spinning clockwise and False if the radar is spinning counter clockwise

    # derivative_set: bool = False  # Is the derivatives calculated
    # d0img: np.ndarray = None  # The simple derivative of the scan in the first direction
    # d1img: np.ndarray = None  # The simple derivative of the scan in the second direction

    def __getitem__(self, item):
        return linear_interpolation(self.img, np.array(item))

    def copyMeta(self):
        return EuclideanScan(sample_size=self.sample_size, range_scales=self.range_scales,
                             range_scale_lengths=self.range_scale_lengths, true_size=self.true_size,
                             timestamp_start_capture=self.timestamp_start_capture,
                             timestamp_end_capture=self.timestamp_end_capture, pose=self.pose, center=self.center,
                             radius=self.radius, img=np.array([]))

    def subsample(self):
        newScan = self.copyMeta()
        newScan.img = cv2.pyrDown(self.img, dstsize=(self.img.shape[0] // 2, self.img.shape[1] // 2))
        newScan.true_size = False
        newScan.sample_size = self.sample_size * 2.0
        return newScan

    def unproject_coordinate(self, pt: np.ndarray):
        """
        Calculates the (x, y) coordinate relative to the ships position from the pixel coordinate in the scan.
        Both values are in meters.
        """
        if self.clockwise:
            orientation = 1
        else:
            orientation = -1
        print("Waring, this function does not work properly.")
        # exit(1)
        rel_pos = pt - self.center
        rel_pos = np.array([-rel_pos[1], orientation * rel_pos[0]])
        dist = np.linalg.norm(rel_pos)
        return rel_pos * (range_from_scaled_lengths(dist, self.range_scales, self.range_scale_lengths,
                                                    self.sample_size) / dist)

    def project_coordinate(self, point_rn_rn_x: np.ndarray):
        if self.clockwise:
            orientation = 1
        else:
            orientation = -1
        img_coord = np.array([orientation * point_rn_rn_x[1], - point_rn_rn_x[0]])
        # (range_m, angle) = self.get_range_angle(point_rn_rn_x)
        dist = np.linalg.norm(img_coord)
        if dist == 0:
            return self.center
        range_px = px_coord_from_scaled_lenghts(dist, self.range_scales, self.range_scale_lengths, self.sample_size)
        img_coord = img_coord * (range_px / dist)
        return img_coord + self.center

    def get_range_angle(self, pt):
        """
        Gets the real world angle and range from the pixel coordinate
        @return (range [m], angle [0-2*pi],)
        """
        print("WARNING: This function is not working correctly. (get_range_angle)")
        # exit(0)
        if self.clockwise:
            orientation = 1
        else:
            orientation = -1
        rel_px_pos = pt - self.center
        angle = orientation * np.arctan2(-rel_px_pos[1], -rel_px_pos[0]) + np.pi / 2
        range_temp = np.linalg.norm(rel_px_pos)
        range_t = range_from_scaled_lengths(range_temp, self.range_scales, self.range_scale_lengths, self.sample_size)
        return (range_t, angle,)

    def calculate_gradients(self):
        # This function is working only with the image gradients, not the real world size gradients.
        dxu = self.img[2:, :]
        dxd = self.img[:-2, :]
        d0img = (dxu - dxd) / 2.0
        dyu = self.img[:, 2:]
        dyd = self.img[:, :-2]
        d1img = (dyu - dyd) / 2.0
        # img = self.img[1:-1, 1:-1]
        # self.derivative_set = True
        return d0img, d1img


@dataclass
class EuclidianPyramid:
    """
    Class to hold the subsampled images of the scan.
    Level 0 is bot_lvl, i.e. level 0 is the original size scan if bot_lvl=0.
    I.e. is there top_lvl - bot_lvl + 1 elements in lvl so that lvl[top_lvl] is valid.
    """
    lvl: List[EuclideanScan]  # The subsampled image pyramid levels.
    top_lvl: int  # The including top level of the image pyramid.
    bot_lvl: int  # the including bottom level of the image pyramid.

    def __getitem__(self, item):
        return self.lvl[item]

    @staticmethod
    def create(scan: EuclideanScan, top_lvl, bot_lvl=0):
        new_pyr = EuclidianPyramid([], top_lvl, bot_lvl)
        lvl_i = scan
        for i in range(top_lvl + 1):
            if i >= bot_lvl:
                new_pyr.lvl.append(lvl_i)
            lvl_i = lvl_i.subsample()
        return new_pyr
