from radar_utils.scan import PolarScan
import numpy as np
# from numba import jit

INTERFERENCE_THRESHOLD = 5


def radar_interference_filter(scan: PolarScan, sliding_window_size=1):
    """
    Warning, this will now alter the input.
    Warning: Work in progress. It does not work well enough.
    :param scan:
    :return:
    """
    img = scan.img[sliding_window_size:-sliding_window_size, :]
    res = radar_interference_filter_internal(scan.img, sliding_window_size)
    scan.img = res.astype(np.uint8)
    return scan

# TODO: Consider numbas fastmath for this.
# @jit(nopython=True, parallel=True)
def radar_interference_filter_internal(img, sliding_window_size):
    sl_window = img[sliding_window_size:-sliding_window_size, :]
    avg = (img[0:-sliding_window_size * 2, :].astype(np.uint16) + img[sliding_window_size * 2:, :]).astype(
        np.uint16) / (2.0)
    diff = (sl_window.astype(np.int16) - avg).astype(np.int16)
    res = np.where(diff > INTERFERENCE_THRESHOLD, avg, sl_window)
    return res


if __name__ == '__main__':
    import radar_utils.viz as viz

    scan = PolarScan.load("/home/henrik/Data/polarlys/2018-06-15-17_41_30/Radar0/2018-06-15-17_41_30.bmp")
    filtered = radar_interference_filter(scan)
    viz.disp_pol(filtered)
