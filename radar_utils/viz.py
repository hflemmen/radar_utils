import os
import numpy as np
from radar_utils.scan import PolarScan, EuclideanScan
import radar_utils.transformations as transformations
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import manifpy as mf
import cv2
import radar_utils.video_creator


def show_scan(scan: PolarScan):
    plt.imshow(scan.img)
    plt.show()


class PolarGui:

    def __init__(self, scan: PolarScan):
        self.scan = scan
        self.ax = plt.imshow(scan.img)
        axpolar = plt.axes([0.78, 0.05, 0.1, 0.075])
        axeuc = plt.axes([0.89, 0.05, 0.1, 0.075])
        bpolar = plt.Button(axpolar, "Polar")
        beuc = plt.Button(axeuc, "Eucledian")
        beuc.on_clicked(self.change_to_euc)
        bpolar.on_clicked(self.reset_to_polar)
        plt.show()

    def change_to_euc(self, event, new_size=None):
        scan = transformations.fillInRanges(self.scan)
        euc = transformations.polarToEuc(scan, new_size)
        self.ax.set_data(euc.img)
        numrows, numcols = euc.img.shape
        self.ax.set_extent(((-0.5, numcols - 0.5, numrows - 0.5, -0.5)))
        plt.draw()

    def reset_to_polar(self, event):
        self.ax.set_data(self.scan.img)
        numrows, numcols = self.scan.img.shape
        self.ax.set_extent(((-0.5, numcols - 0.5, numrows - 0.5, -0.5)))
        plt.draw()


def disp_pol(scan: PolarScan):
    gui = PolarGui(scan)


def disp_euc(scan: EuclideanScan):
    plt.imshow(scan.img)
    plt.show()


@dataclass
class UpdateStepData:
    scan_rn: PolarScan
    est_Ts_r0_rn: List[mf.SE2]
    gt_Ts_w_gn: List[mf.SE2]
    landmarks_r0: 'List[np.ndarray[2]]'
    removed: 'np.ndarray[int]'
    landmarks_r0_t1: 'List[np.ndarray[2]]'
    joint_covariance: 'Dict[np.ndarray[5,5]]'


def draw_points(mask, landmarks_r0, scan, est_T_r0_rn: mf.SE2, color, window_size,
                jointCovariance: 'Dict[int, np.ndarray[5,5]]' = None, active_landmarks=None):
    for i, point_r0_r0_lx in enumerate(landmarks_r0):
        point_rn_rn_lx = est_T_r0_rn.inverse().act(point_r0_r0_lx)
        try:
            corrected_this = scan.project_coordinate(point_rn_rn_lx)
        except ValueError as e:
            # The coordinate got backprojected to outside of the image
            continue
        a, b = corrected_this[0], corrected_this[1]
        if jointCovariance is None:
            cv2.circle(mask, (int(a), int(b)), 20, color, -1)
        else:
            cov_x_li = jointCovariance[i]
            Pll = cov_x_li[:2, :2]
            Pxx = cov_x_li[2:, 2:]
            Pxl = cov_x_li[2:, :2]
            Pl_cond_x = Pll - Pxl.T @ np.linalg.inv(Pxx) @ Pxl
            # We use the transformed coordinate even if this is not proper conditining.

            v, w = np.linalg.eig(Pl_cond_x)
            angle = np.arctan2(w[1][1], w[1][0])
            scaled_v = np.sqrt(v) / scan.range_scales[
                0]  # Does not multiply by two since we only give in half the axis.
            mask = cv2.ellipse(mask, (int(a), int(b),), (int(scaled_v[1]), int(scaled_v[0]),), angle * 180.0 / np.pi, 0,
                               360, color, 5,
                               cv2.LINE_8)


error_int_interpretation = ['KLT failed', 'Backtrack failed', 'Both failed']


def draw_removed_landmarks(error_int, color, data, est_T_r0_rn, eucScan, mask, window_size):
    removed_landmarks_st = []
    for index, i in enumerate(data.removed.flatten()):
        if i == error_int:
            removed_landmarks_st.append(data.landmarks_r0_t1[index])
    if len(removed_landmarks_st) > 0:
        draw_points(mask, removed_landmarks_st, eucScan, est_T_r0_rn, color, window_size)
    return removed_landmarks_st


def draw_legend(mask: np.ndarray, annotations: List[str], colors: List[Tuple[int, int, int]]):
    for i, (anotation, color) in enumerate(zip(annotations, colors)):
        cv2.circle(mask, (30, 40 * (i + 1)), 20, color, -1)
        cv2.putText(mask, str(anotation), (55, int(40 * (i + 1.4))), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)


class TrajectoryVisualizer:

    def __init__(self, T_w_g0: mf.SE2, T_g0_r0: mf.SE2, path, show_visualisation, record_video, use_cache=False):
        self.T_w_g0 = T_w_g0
        self.T_g0_r0 = T_g0_r0
        self.video_sink = None
        self.path = path
        self.show_visualisation = show_visualisation
        self.record_video = record_video
        self.use_cache = use_cache

    def viz_trajectory(self, data: UpdateStepData) -> np.ndarray:
        window_size = 4000
        eucScan = transformations.polarToEuc(data.scan_rn, np.array([window_size, window_size]), self.use_cache)
        # vizImg = eucScan.img
        # color = np.random.randint(0, 255, (1000, 3), dtype='uint8')
        mask = np.zeros((eucScan.img.shape[0], eucScan.img.shape[1], 3), dtype=np.uint8)
        self.draw_est_trajectory(mask, data, eucScan, window_size, (255, 0, 0))
        self.draw_gt_trajectory(mask, data, eucScan, window_size, (0, 255, 0))

        est_T_r0_rn = data.est_Ts_r0_rn[-1]
        draw_points(mask, data.landmarks_r0, eucScan, est_T_r0_rn, (0, 0, 255), window_size, data.joint_covariance)

        draw_removed_landmarks(1, (255, 0, 0), data, est_T_r0_rn, eucScan, mask, window_size)

        draw_removed_landmarks(2, (0, 255, 0), data, est_T_r0_rn, eucScan, mask, window_size)

        draw_removed_landmarks(3, (255, 255, 0), data, est_T_r0_rn, eucScan, mask, window_size)

        draw_legend(mask, error_int_interpretation, [(255, 0, 0), (0, 255, 0), (255, 255, 0)])

        drawn = cv2.add(cv2.cvtColor(eucScan.img, cv2.COLOR_GRAY2BGR), mask)
        viz = cv2.resize(drawn, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

        viz_rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
        if self.record_video:
            if self.video_sink is None:
                # os.makedirs(self.path, exist_ok=True)
                self.video_sink = radar_utils.video_creator.VideoSink(self.path + "trajectory", 10)
            self.video_sink.add(viz_rgb)
        if self.show_visualisation:
            cv2.imshow("Trajectory", viz_rgb)
        vis_szcan = data.scan_rn.copyMeta()
        vis_szcan.img = viz_rgb
        disp_pol(vis_szcan)
        return viz_rgb

    def draw_gt_trajectory(self, mask, data: UpdateStepData, scan: EuclideanScan, window_size: int,
                           color: Tuple[int, int, int]):
        for i in range(len(data.gt_Ts_w_gn) - 1):
            p_w_w_gi = data.gt_Ts_w_gn[i].translation()
            p_w_w_gi1 = data.gt_Ts_w_gn[i + 1].translation()
            T_rn_w = data.est_Ts_r0_rn[-1].inverse() * self.T_g0_r0.inverse() * self.T_w_g0.inverse()
            p_rn_rn_gi = T_rn_w.act(p_w_w_gi)
            p_rn_rn_gi1 = T_rn_w.act(p_w_w_gi1)
            try:
                corrected_this = scan.project_coordinate(p_rn_rn_gi)
                corrected_next = scan.project_coordinate(p_rn_rn_gi1)
            except ValueError:
                continue
            a, b = corrected_this[0], corrected_this[1]
            c, d = corrected_next[0], corrected_next[1]
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 10)

    def draw_est_trajectory(self, mask, data: UpdateStepData, scan: EuclideanScan, window_size: int,
                            color: Tuple[int, int, int]):
        Ts_r0_ri = data.est_Ts_r0_rn
        est_T_r0_rn = data.est_Ts_r0_rn[-1]
        for i in range(len(Ts_r0_ri) - 1):
            p_r0_r0_ri = Ts_r0_ri[i].translation()
            p_r0_r0_ri1 = Ts_r0_ri[i + 1].translation()
            p_rn_rn_ri = est_T_r0_rn.inverse().act(p_r0_r0_ri)
            p_rn_rn_ri1 = est_T_r0_rn.inverse().act(p_r0_r0_ri1)
            try:
                corrected_this = scan.project_coordinate(p_rn_rn_ri)
                corrected_next = scan.project_coordinate(p_rn_rn_ri1)
            except ValueError:
                continue
            a, b = corrected_this[0], corrected_this[1]
            c, d = corrected_next[0], corrected_next[1]
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 10)

    def release(self):
        if self.record_video:
            self.video_sink.release()


color = np.random.randint(0, 255, (1000, 3), dtype='uint8')


def draw_tracks(feature_tracks, current_active, mask, text_scale=1):
    """
    Draws feature tracks, which are on the same format used by the optical flow tracker.
    It assumes that the feature coordinates are given in pixel coordinates, relative to the given mask.
    """
    # mask = np.zeros(shape, dtype='uint8')
    hist = 30  # 4
    for id in current_active:
        track = feature_tracks[id]
        for i in range(1, hist):
            if i + 1 > len(track):
                break
            a, b = track[-i][0], track[-i][1]
            c, d = track[-i - 1][0], track[-i - 1][1]
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[id % 1000].tolist(), 5 * text_scale)
        e, f = track[-1][0], track[-1][1]
        # mask = cv2.circle(mask, (int(e), int(f)), 25, color[id % 1000].tolist(), -1)
        mask = cv2.putText(mask, str(id), (int(e), int(f)), cv2.FONT_HERSHEY_SIMPLEX, 4 * text_scale,
                           color[id % 1000].tolist(),  # (255, 0, 255),
                           3 * text_scale)
    return mask


@dataclass
class StateCovariances:
    landmark_cov: 'Dict[int, np.ndarray[2,2]]'
    pose_cov: 'List[np.ndarray[3, 3]]'
    # The joint distribution over the most recent state and each observed landmark
    # [P(Li), -]
    # [-, P(Xn)]
    joint_cov_rn: 'List[np.ndarray[5, 5]]'


def draw_ellipses(marginals: StateCovariances, feature_tracks, current_active, mask):
    for id in current_active:
        track = feature_tracks[id][-1]
        cov = marginals.landmark_cov[id]
        v, w = np.linalg.eig(cov)
        angle = np.arctan2(w[1][1], w[1][0])
        scaled_v = np.sqrt(v)  # Does not multiply by two since we only give in half the axis.
        # mask = cv2.ellipse(mask, track.astype(int), (int(scaled_v[1]), int(scaled_v[0]),), angle * 180.0 / np.pi, 0,
        #                    360, color[id % 1000].tolist(), 20,
        #                    cv2.LINE_8)
        mask = cv2.circle(mask, (int(track[0]), int(track[1])), 25, color[id % 1000].tolist(), -1)


@dataclass
class CartUpdateData:
    landmarks_r0: "List[np.ndarray[2]]"
    current_ids: List[int]
    est_Ts_r0_rn: List[mf.SE2]
    marginals: StateCovariances


def transform_w_cart_img(p_w: "np.ndarray[2]", center_img: "np.ndarray[2]"):
    """
    Transforms a NED-position to the image coordinate frame.
    Center of the image is the NED-position (0, 0)
    The image frame is aligned to
    p: (North, East)
    return: (Right, Down)
    """
    res = np.array([p_w[1], -p_w[0], ]) + center_img
    return res


class CartesianVisualizer:

    def __init__(self, T_w_g0: mf.SE2, T_g0_r0: mf.SE2, stable_window_size: int, show_visualization: bool,
                 record_video: bool):
        self.stable_feature_tracks_w_w_lx = {}
        self.T_g0_r0 = T_g0_r0
        self.T_w_g0 = T_w_g0
        self.stable_window_size = stable_window_size
        self.show_visualisation = show_visualization
        self.record_video = record_video  # Currently unused

    def update(self, data: CartUpdateData) -> np.ndarray:
        center_of_image = np.array([self.stable_window_size / 2, self.stable_window_size / 2])

        # Calculate feature position
        T_w_r0 = self.T_w_g0 * self.T_g0_r0
        for i in range(len(data.landmarks_r0)):
            id = data.current_ids[i]
            if id not in self.stable_feature_tracks_w_w_lx:
                self.stable_feature_tracks_w_w_lx[id] = []
            self.stable_feature_tracks_w_w_lx[id].append(
                transform_w_cart_img(T_w_r0.act(data.landmarks_r0[i]), center_of_image))

        # Draw features
        shape = (self.stable_window_size, self.stable_window_size, 3)
        mask = np.zeros(shape, dtype='uint8')
        mask = draw_tracks(self.stable_feature_tracks_w_w_lx, data.current_ids, mask, text_scale=4)
        ajusted_pos = None
        # Draw feature ellipses
        if data.marginals is not None:
            draw_ellipses(data.marginals, self.stable_feature_tracks_w_w_lx, data.current_ids, mask)

        # Draw ship path
        for i in range(1, len(data.est_Ts_r0_rn)):
            T_w_ri = T_w_r0 * data.est_Ts_r0_rn[i]
            T_w_ri1 = T_w_r0 * data.est_Ts_r0_rn[i - 1]
            ajusted_pos = transform_w_cart_img(T_w_ri.translation(), center_of_image)
            ajusted_old_pos = transform_w_cart_img(T_w_ri1.translation(), center_of_image)
            cv2.line(mask, (int(ajusted_pos[0]), int(ajusted_pos[1])),
                     (int(ajusted_old_pos[0]), int(ajusted_old_pos[1])), [0, 0, 255], 70)

        # Draw ship
        est_T_w_rn = T_w_r0 * data.est_Ts_r0_rn[-1]
        ajusted_pos_start = transform_w_cart_img(est_T_w_rn.translation(), center_of_image)

        arrowsize = 100
        pos_end = est_T_w_rn.translation() + arrowsize * 10 * np.array(
            [np.cos(est_T_w_rn.angle()), np.sin(est_T_w_rn.angle())])
        ajusted_pos_end = transform_w_cart_img(pos_end, center_of_image)

        cv2.arrowedLine(mask, ajusted_pos_start.astype(np.int32).tolist(),
                        ajusted_pos_end.astype(np.int32).tolist(), [0, 0, 255], arrowsize, tipLength=0.2)

        drawn_viz = cv2.resize(mask, None, fx=0.07, fy=0.07, interpolation=cv2.INTER_LINEAR)
        if self.show_visualisation:
            cv2.imshow("NED plot", drawn_viz)
        return drawn_viz


@dataclass
class FeatureVisualizerData:
    img: np.ndarray
    features: List[np.ndarray]
    active_features: List[int]
    removed: 'np.ndarray[int]'


class FeatureVisualizer:
    """
    Opencv has a fixed size window over all the pyramid levels, which
     means that they cover a larger area for the higher ones.
    """

    def __init__(self, sw_cfg, hw_cfg):
        self.sw_cfg = sw_cfg
        self.hw_cfg = hw_cfg
        self.store_patches = True
        self.old_active_features = None

    def update(self, data: FeatureVisualizerData):
        if not (self.sw_cfg.show_visualisation):  # or self.sw_cfg.record_video):
            return
        patches = []
        indices = []
        window_size_half = int(self.sw_cfg.klt_window_size / 2) * 2 ** self.sw_cfg.klt_max_pyr_level
        for i, f in data.features.items():
            if i in data.active_features:
                f = f[-1]

                patch = data.img[int(f[1]) - window_size_half:int(f[1]) + 1 + window_size_half,
                        int(f[0]) - window_size_half:int(f[0]) + 1 + window_size_half]
                if patch.shape[0] <= 0 or patch.shape[1] <= 0:
                    continue
                patches.append(patch)
                indices.append(i)

                # Store patch
                if self.store_patches:
                    patches_path = self.sw_cfg.get_iteration_path(self.hw_cfg.in_path) + "Patches/"
                    spesific_patch_path = patches_path + f"patch{i}/"
                    os.makedirs(spesific_patch_path, exist_ok=True)
                    old_patches = os.listdir(spesific_patch_path)

                    cv2.imwrite(spesific_patch_path + f"iter{len(old_patches) + 1}.png", patch)

        window_size = window_size_half * 2
        columns = 5
        total_rows = len(patches) // columns + 1
        img = np.zeros((total_rows * (window_size + 1), columns * (window_size + 1)), dtype=np.uint8)
        i = 0
        for x0 in range(total_rows):
            for x1 in range(columns):
                if i < len(patches):
                    img[x0 * window_size:(x0 * window_size) + patches[i].shape[0],
                    (x1 * window_size):(x1 * window_size) + patches[i].shape[1]] = patches[i]
                    i += 1
                else:
                    break

        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_c = cv2.pyrDown(img_c)

        if self.sw_cfg.show_visualisation:
            cv2.imshow("patches", img_c)

        # Write out deleted patches
        if self.store_patches and self.old_active_features is not None:
            for index, i in enumerate(data.removed.flatten()):
                feature_nr = self.old_active_features[index]
                if i != 0:
                    patches_path = self.sw_cfg.get_iteration_path(self.hw_cfg.in_path) + "Patches/"
                    spesific_patch_path = patches_path + f"patch{feature_nr}/"
                    os.makedirs(spesific_patch_path, exist_ok=True)
                    old_patches = os.listdir(spesific_patch_path)
                    with open(spesific_patch_path + f"iter{len(old_patches) + 1}_death.txt", 'w') as f:
                        f.write(f"Died from {error_int_interpretation[i - 1]}")
        self.old_active_features = data.active_features.copy()

    def release(self):
        pass


def main():
    scan = PolarScan.load("/home/henrik/Data/polarlys/2018-06-20-20_05_30/Radar0/2018-06-20-20_05_40.bmp")
    disp_pol(scan)


if __name__ == '__main__':
    main()
