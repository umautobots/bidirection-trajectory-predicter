import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, quantile_transform
from .trajectory_ops import compute_bounding_box
import pdb

class Trajectory:
    def __init__(self, trajectory_id, frames, coordinates, resolution):
        self.trajectory_id = trajectory_id
        # self.person_id = trajectory_id.split('_')[1]
        # NOTE: remove the missing steps in front and rear 
        non_zero_indices = np.where(coordinates.max(axis=1) > 0)[0]
        if len(non_zero_indices) == 0:
            self.coordinates = []
            self.frames = []
        else:
            start = non_zero_indices[0]
            end = non_zero_indices[-1]+1
            self.coordinates = coordinates[start:end]
            self.frames = frames[start:end]
    
        
        self.is_global = False
        self.resolution = resolution # (width, height)
    def __len__(self):
        return len(self.frames)

    def use_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        self.coordinates = self._extract_global_features(video_resolution=video_resolution, extract_delta=extract_delta,
                                                         use_first_step_as_reference=use_first_step_as_reference)
        self.is_global = True

    def use_size_features(self, video_resolution):
        self.coordinates = self._extract_size_features(video_resolution=video_resolution)

    def _extract_size_features(self, video_resolution):
        bbs = np.apply_along_axis(compute_bounding_box, axis=1, arr=self.coordinates, video_resolution=video_resolution)
        bbs_measures = np.apply_along_axis(self._extract_bounding_box_measurements, axis=1, arr=bbs)
        return bbs_measures

    def _extract_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=self.coordinates,
                                             video_resolution=video_resolution)
        bbs_measures = np.apply_along_axis(self._extract_bounding_box_measurements, axis=1, arr=bounding_boxes)
        bbs_centre = np.apply_along_axis(self._extract_bounding_box_centre, axis=1, arr=bounding_boxes)
        if extract_delta:
            bbs_delta = np.vstack((np.full((1, 2), fill_value=1e-7), np.diff(bbs_centre, axis=0)))

        if use_first_step_as_reference:
            bbs_centre -= bbs_centre[0]
            # bbs_centre /= np.where(bbs_measures == 0.0, 1.0, bbs_measures)
            bbs_centre[0] += 1e-6

        if extract_delta:
            return np.hstack((bbs_centre, bbs_delta, bbs_measures))

        return np.hstack((bbs_centre, bbs_measures))

    @staticmethod
    def _extract_bounding_box_centre(bb):
        x = (bb[0] + bb[1]) / 2
        y = (bb[2] + bb[3]) / 2

        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _extract_bounding_box_measurements(bb):
        width = bb[1] - bb[0]
        height = bb[3] - bb[2]

        return np.array([width, height], dtype=np.float32)

    def change_coordinate_system(self, video_resolution, coordinate_system='global', invert=False):
        if invert:
            if coordinate_system == 'global':
                self.coordinates = self._from_global_to_image(self.coordinates, video_resolution=video_resolution)
            else:
                raise ValueError('Unknown coordinate system. Only global is available for inversion.')
        else:
            if coordinate_system == 'global':
                self.coordinates = self._from_image_to_global(self.coordinates, video_resolution=video_resolution)
            elif coordinate_system == 'bounding_box_top_left':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='top_left')
            elif coordinate_system == 'bounding_box_centre':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='centre')
            else:
                raise ValueError('Unknown coordinate system. Please select one of: global, bounding_box_top_left, or '
                                 'bounding_box_centre.')

    @staticmethod
    def _from_global_to_image(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) * video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_global(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) / video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_bounding_box(coordinates, video_resolution, location='centre'):
        if location == 'top_left':
            fn = Trajectory._from_image_to_top_left_bounding_box
        elif location == 'centre':
            fn = Trajectory._from_image_to_centre_bounding_box
        else:
            raise ValueError('Unknown location for the bounding box. Please select either top_left or centre.')

        coordinates = fn(coordinates, video_resolution=video_resolution)

        return coordinates

    @staticmethod
    def _from_image_to_top_left_bounding_box(coordinates, video_resolution):
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, float(left), xs), np.where(ys == 0.0, float(top), ys)
                xs, ys = (xs - left) / (right - left), (ys - top) / (bottom - top)
                kps = np.hstack((xs, ys)).ravel()

            coordinates[idx] = kps

        return coordinates

    @staticmethod
    def _from_image_to_centre_bounding_box(coordinates, video_resolution):
        # TODO: Better implementation
        # coordinates = np.where(coordinates == 0, np.nan, coordinates)
        # bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=coordinates,
        #                                      video_resolution=video_resolution)
        # centre_x = (bounding_boxes[:, 0] + bounding_boxes[:, 1]) / 2
        # centre_y = (bounding_boxes[:, 2] + bounding_boxes[:, 3]) / 2
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, centre_x, xs) - centre_x, np.where(ys == 0.0, centre_y, ys) - centre_y
                left, right, top, bottom = left - centre_x, right - centre_x, top - centre_y, bottom - centre_y
                width, height = right - left, bottom - top
                if width < 1 or height < 1:
                    pdb.set_trace()
                if (xs/width).max()>10:
                    pdb.set_trace()
                xs, ys = xs / width, ys / height
                kps = np.hstack((xs, ys)).ravel()
                
            coordinates[idx] = kps

        return coordinates

    def is_short(self, input_length, input_gap, pred_length=0):
        min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

        return len(self) < min_trajectory_length

    def input_missing_steps(self):
        """Fill missing steps with a weighted average of the closest non-missing steps."""
        trajectory_length, input_dim = self.coordinates.shape
        last_step_non_missing = 0
        consecutive_missing_steps = 0
        while last_step_non_missing < trajectory_length - 1:
            step_is_missing = np.sum(self.coordinates[last_step_non_missing + 1, :] == 0) == input_dim
            while step_is_missing:
                consecutive_missing_steps += 1
                if last_step_non_missing + 1 + consecutive_missing_steps < trajectory_length:
                    step_is_missing = \
                        np.sum(self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :] == 0) == input_dim
                else:
                    # NOTE: the rest of the trajectory is all missed. do not propagate
                    consecutive_missing_steps = 0
                    break

            if consecutive_missing_steps:
                start_trajectory = self.coordinates[last_step_non_missing, :]
                end_trajectory = self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :]
                for n in range(1, consecutive_missing_steps + 1):
                    a = ((consecutive_missing_steps + 1 - n) / (consecutive_missing_steps + 1)) * start_trajectory
                    b = (n / (consecutive_missing_steps + 1)) * end_trajectory
                    fill_step = a + b
                    fill_step = np.where((start_trajectory == 0) | (end_trajectory == 0), 0, fill_step)
                    self.coordinates[last_step_non_missing + n, :] = fill_step

            last_step_non_missing += consecutive_missing_steps + 1
            consecutive_missing_steps = 0
        
        

