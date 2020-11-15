import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, quantile_transform
import copy
import pdb

def inverse_scale_and_coord(video_resolution, merged_output, out_scaler=None):
    '''inverse scale and coord for y_merged or y_global'''
    merged_output = copy.deepcopy(merged_output)
    merged_output = np.where(merged_output == 0.0, np.nan, merged_output)
    if out_scaler is not None:
        merged_output = inverse_scale(merged_output, scaler=out_scaler)
    merged_output = restore_global_coordinate_system(merged_output, video_resolution=video_resolution)
    merged_output = np.where(np.isnan(merged_output), 0.0, merged_output)
    return merged_output

def inverse_scale(X, scaler):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)
    X = scaler.inverse_transform(X)
    X = X.reshape(original_shape)
    return X
    
def restore_global_coordinate_system(X, video_resolution):
    '''restore global coordinate for y_merged or y_global'''
    original_shape = X.shape
    # X = X.reshape(-1, 2) * video_resolution
    # X = X.reshape(original_shape)
    X = X.reshape(X.shape[0], X.shape[1], -1, 2) * video_resolution[:, None, None, :]
    X = X.reshape(original_shape)
    return X

def aggregate_rnn_autoencoder_data(trajectories, input_length, input_gap=0, pred_length=0):
    Xs, Xs_pred, Xs_traj_id, Xs_frame, Xs_resolutions = [], [], [], [], []
    
    for traj_id, trajectory in trajectories.items():
        X, X_pred, X_frame = _aggregate_rnn_autoencoder_data(trajectory, input_length, input_gap, pred_length)
        Xs.append(X)
        Xs_frame.append(X_frame)
        Xs_traj_id += [traj_id for i in range(len(X_frame))]
        Xs_resolutions += [trajectory.resolution for i in range(len(X_frame))]
        if X_pred is not None:
            Xs_pred.append(X_pred)
    
    Xs = np.vstack(Xs)
    Xs_frame = np.vstack(Xs_frame)
    if not Xs_pred:
        Xs_pred = None
    else:
        Xs_pred = np.vstack(Xs_pred)

    return Xs, Xs_pred, Xs_frame, Xs_traj_id, Xs_resolutions


def _aggregate_rnn_autoencoder_data(trajectory, input_length, input_gap=0, pred_length=0):
    coordinates = trajectory.coordinates
    frames = trajectory.frames
    input_trajectories, future_trajectories = [], None
    input_frames = []
    total_input_seq_len = input_length + input_gap * (input_length - 1)
    step = input_gap + 1
    if pred_length > 0:
        future_trajectories = []
        stop = len(coordinates) - pred_length - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
            input_frames.append(frames[start_index:stop_index:step])
            future_trajectories.append(coordinates[stop_index:(stop_index + pred_length), :])
        input_trajectories = np.stack(input_trajectories, axis=0)
        input_frames = np.stack(input_frames, axis=0)
        future_trajectories = np.stack(future_trajectories, axis=0)
    else:
        stop = len(coordinates) - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
            input_frames.append(frames[start_index:stop_index:step])
        input_trajectories = np.stack(input_trajectories, axis=0)
        input_frames = np.stack(input_frames, axis=0)
    return input_trajectories, future_trajectories, input_frames

def aggregate_rnn_ae_evaluation_data(trajectories, input_length, input_gap, pred_length, overlapping_trajectories):
    trajectories_ids, frames, X = [], [], []
    for trajectory in trajectories.values():
        traj_ids, traj_frames, traj_X = _aggregate_rnn_ae_evaluation_data(trajectory, input_length)
        trajectories_ids.append(traj_ids)
        frames.append(traj_frames)
        X.append(traj_X)

    trajectories_ids, frames, X = np.vstack(trajectories_ids), np.vstack(frames), np.vstack(X)

    return trajectories_ids, frames, X


def _aggregate_rnn_ae_evaluation_data(trajectory, input_length):
    traj_frames, traj_X = [], []
    coordinates = trajectory.coordinates
    frames = trajectory.frames

    total_input_seq_len = input_length
    stop = len(coordinates) - total_input_seq_len + 1
    for start_index in range(stop):
        stop_index = start_index + total_input_seq_len
        traj_X.append(coordinates[start_index:stop_index, :])
        traj_frames.append(frames[start_index:stop_index])
    traj_frames, traj_X = np.stack(traj_frames, axis=0), np.stack(traj_X, axis=0)

    trajectory_id = trajectory.trajectory_id
    traj_ids = np.full(traj_frames.shape, fill_value=trajectory_id)

    return traj_ids, traj_frames, traj_X

def remove_short_trajectories(trajectories, input_length, input_gap, pred_length=0):
    filtered_trajectories = {}
    for trajectory_id, trajectory in trajectories.items():
        if not trajectory.is_short(input_length=input_length, input_gap=input_gap, pred_length=pred_length):
            filtered_trajectories[trajectory_id] = trajectory

    return filtered_trajectories

def split_into_train_and_test(trajectories, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    trajectories_ids = []
    trajectories_lengths = []
    for trajectory_id, trajectory in trajectories.items():
        trajectories_ids.append(trajectory_id)
        trajectories_lengths.append(len(trajectory))

    sorting_indices = np.argsort(trajectories_lengths)
    q1_idx = round(len(sorting_indices) * 0.25)
    q2_idx = round(len(sorting_indices) * 0.50)
    q3_idx = round(len(sorting_indices) * 0.75)

    sorted_ids = np.array(trajectories_ids)[sorting_indices]
    train_ids = []
    val_ids = []
    quantiles_indices = [0, q1_idx, q2_idx, q3_idx, len(sorting_indices)]
    for idx, q_idx in enumerate(quantiles_indices[1:], 1):
        q_ids = sorted_ids[quantiles_indices[idx - 1]:q_idx]
        q_ids = np.random.permutation(q_ids)
        train_idx = round(len(q_ids) * train_ratio)
        train_ids.extend(q_ids[:train_idx])
        val_ids.extend(q_ids[train_idx:])

    trajectories_train = {}
    for train_id in train_ids:
        trajectories_train[train_id] = trajectories[train_id]

    trajectories_val = {}
    for val_id in val_ids:
        trajectories_val[val_id] = trajectories[val_id]

    return trajectories_train, trajectories_val

def compute_bounding_box(keypoints, video_resolution, return_discrete_values=True):
    """Compute the bounding box of a set of keypoints.
    Argument(s):
        keypoints -- A numpy array, of shape (num_keypoints * 2,), containing the x and y values of each
            keypoint detected.
        video_resolution -- A numpy array, of shape (2,) and dtype float32, containing the width and the height of
            the video.
    Return(s):
        The bounding box of the keypoints represented by a 4-uple of integers. The order of the corners is: left,
        right, top, bottom.
    """
    width, height = video_resolution
    keypoints_reshaped = keypoints.reshape(-1, 2)
    x, y = keypoints_reshaped[:, 0], keypoints_reshaped[:, 1]
    x, y = x[x != 0.0], y[y != 0.0]
    try:
        left, right, top, bottom = np.min(x), np.max(x), np.min(y), np.max(y)
    except ValueError:
        # print('All joints missing for input skeleton. Returning zeros for the bounding box.')
        return 0, 0, 0, 0

    extra_width, extra_height = 0.1 * (right - left + 1), 0.1 * (bottom - top + 1)
    left, right = np.clip(left - extra_width, 0, width - 1), np.clip(right + extra_width, 0, width - 1)
    top, bottom = np.clip(top - extra_height, 0, height - 1), np.clip(bottom + extra_height, 0, height - 1)
    # left, right = left - extra_width, right + extra_width
    # top, bottom = top - extra_height, bottom + extra_height

    if return_discrete_values:
        left = int(round(left))
        right = max([left + 1, int(round(right))])
        top = int(round(top))
        bottom = max([top + 1, int(round(bottom))])
        return left, right, top, bottom
    else:
        return left, right, top, bottom

def extract_global_features(trajectories, extract_delta=False, use_first_step_as_reference=False):
    for trajectory in trajectories.values():
        trajectory.use_global_features(video_resolution=trajectory.resolution, extract_delta=extract_delta,
                                       use_first_step_as_reference=use_first_step_as_reference)

    return trajectories


def extract_size_features(trajectories):
    for trajectory in trajectories.values():
        trajectory.use_size_features(video_resolution=trajectory.resolution)

    return trajectories


def change_coordinate_system(trajectories, coordinate_system='global', invert=False):
    for trajectory in trajectories.values():
        trajectory.change_coordinate_system(trajectory.resolution, coordinate_system=coordinate_system, invert=invert)

    return trajectories


def split_into_train_and_test(trajectories, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    trajectories_ids = []
    trajectories_lengths = []
    for trajectory_id, trajectory in trajectories.items():
        trajectories_ids.append(trajectory_id)
        trajectories_lengths.append(len(trajectory))

    sorting_indices = np.argsort(trajectories_lengths)
    q1_idx = round(len(sorting_indices) * 0.25)
    q2_idx = round(len(sorting_indices) * 0.50)
    q3_idx = round(len(sorting_indices) * 0.75)

    sorted_ids = np.array(trajectories_ids)[sorting_indices]
    train_ids = []
    val_ids = []
    quantiles_indices = [0, q1_idx, q2_idx, q3_idx, len(sorting_indices)]
    for idx, q_idx in enumerate(quantiles_indices[1:], 1):
        q_ids = sorted_ids[quantiles_indices[idx - 1]:q_idx]
        q_ids = np.random.permutation(q_ids)
        train_idx = round(len(q_ids) * train_ratio)
        train_ids.extend(q_ids[:train_idx])
        val_ids.extend(q_ids[train_idx:])

    trajectories_train = {}
    for train_id in train_ids:
        trajectories_train[train_id] = trajectories[train_id]

    trajectories_val = {}
    for val_id in val_ids:
        trajectories_val[val_id] = trajectories[val_id]

    return trajectories_train, trajectories_val


def scale_trajectories(X, scaler=None, strategy='zero_one'):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)
    if strategy == 'zero_one':
        X_scaled, scaler = scale_trajectories_zero_one(X, scaler=scaler)
    elif strategy == 'three_stds':
        X_scaled, scaler = scale_trajectories_three_stds(X, scaler=scaler)
    elif strategy == 'robust':
        X_scaled, scaler = scale_trajectories_robust(X, scaler=scaler)
    elif strategy == 'none':
        X_scaled = X
        scaler = None
    else:
        raise ValueError('Unknown strategy. Please select either zero_one or three_stds.')

    X, X_scaled = X.reshape(original_shape), X_scaled.reshape(original_shape)

    return X_scaled, scaler


def scale_trajectories_zero_one(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)
        X_min = np.nanmin(X, axis=0, keepdims=True)
        X_min = np.where(np.isnan(X_min), 0.0, X_min)
        X_min = np.tile(X_min, reps=[X.shape[0], 1])

        eps = 1e-3
        X = np.where(np.isnan(X), X_min - eps, X)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

    num_examples = X.shape[0]
    X_scaled = np.where(X == 0.0, np.tile(scaler.data_min_, reps=[num_examples, 1]), X)
    X_scaled = scaler.transform(X_scaled)

    return X_scaled, scaler


def scale_trajectories_three_stds(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)

        scaler = StdScaler(stds=3)
        scaler.fit(X)

    X_scaled = np.where(X == 0.0, np.nan, X)
    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def scale_trajectories_robust(X, scaler=None):
    X_scaled = np.where(X == 0.0, np.nan, X)
    if scaler is None:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X_scaled)

    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def aggregate_autoencoder_data(trajectories):
    X = []
    for trajectory in trajectories.values():
        X.append(trajectory.coordinates)

    return np.vstack(X)


def aggregate_autoencoder_evaluation_data(trajectories):
    trajectories_ids, frames, X = [], [], []
    for trajectory_id, trajectory in trajectories.items():
        frames.append(trajectory.frames)
        X.append(trajectory.coordinates)
        trajectories_ids.append(np.repeat(trajectory_id, repeats=len(trajectory.frames)))

    return np.concatenate(trajectories_ids), np.concatenate(frames), np.vstack(X)


def remove_missing_skeletons(X, *arrs):
    non_missing_skeletons = np.sum(np.abs(X), axis=1) > 0.0
    X = X[non_missing_skeletons]
    filtered_arrs = []
    for idx, arr in enumerate(arrs):
        filtered_arrs.append(arr[non_missing_skeletons])

    return X, filtered_arrs


def compute_ae_reconstruction_errors(X, reconstructed_X, loss):
    loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error, 'mse': mean_squared_error}[loss]
    return loss_fn(X, reconstructed_X)


def load_anomaly_masks(anomaly_masks_path):
    file_names = os.listdir(anomaly_masks_path)
    masks = {}
    for file_name in file_names:
        full_id = file_name.split('.')[0]
        file_path = os.path.join(anomaly_masks_path, file_name)
        masks[full_id] = np.load(file_path)

    return masks


def assemble_ground_truth_and_reconstructions(anomaly_masks, trajectory_ids,
                                              reconstruction_frames, reconstruction_errors,
                                              return_video_ids=False):
    y_true, y_hat = {}, {}
    for full_id in anomaly_masks.keys():
        _, video_id = full_id.split('_')
        y_true[video_id] = anomaly_masks[full_id].astype(np.int32)
        y_hat[video_id] = np.zeros_like(y_true[video_id], dtype=np.float32)

    unique_ids = np.unique(trajectory_ids)
    for trajectory_id in unique_ids:
        video_id, _ = trajectory_id.split('_')
        indices = trajectory_ids == trajectory_id
        frames = reconstruction_frames[indices]
        y_hat[video_id][frames] = np.maximum(y_hat[video_id][frames], reconstruction_errors[indices])

    y_true_, y_hat_, video_ids = [], [], []
    for video_id in sorted(y_true.keys()):
        y_true_.append(y_true[video_id])
        y_hat_.append(y_hat[video_id])
        video_ids.extend([video_id] * len(y_true_[-1]))

    y_true_, y_hat_ = np.concatenate(y_true_), np.concatenate(y_hat_)

    if return_video_ids:
        return y_true_, y_hat_, video_ids
    else:
        return y_true_, y_hat_


def quantile_transform_errors(y_hats):
    for camera_id, y_hat in y_hats.items():
        y_hats[camera_id] = quantile_transform(y_hat.reshape(-1, 1)).reshape(-1)

    return y_hats


def input_trajectories_missing_steps(trajectories):
    for trajectory in trajectories.values():
        trajectory.input_missing_steps()

    return trajectories