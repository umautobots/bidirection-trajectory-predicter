import collections
from datetime import datetime
import json
import logging
import numbers
import os
import random
import string
import time

import coloredlogs
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
import torch
import wandb
import pdb
BACKENDS = ('wandb', 'tensorboardx')
RUN_ID_LENGTH = 8


class Logger(logging.Logger):
    """Logger utility. Provides a wrapper over the built-in Python logging utility as
    well as optional visualization backends.

    Parameters
    ----------
    config: Union[dict, OrderedDict]
        A dictionary containing configuration parameters.

    project: str
        Name for the project for a set of runs

    viz_backend: str, default: None
        Backend for visualization and logging.
        Available visualization backends are specified in global BACKENDS.

    sync: bool, default: True
        Whether to sync data to cloud. (Only available for WandB logging at the moment).
    """

    def __init__(self, name, config, project, viz_backend=None, sync=True):
        super().__init__(name)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                coloredlogs.install(fmt='%(asctime)s %(hostname)s %(levelname)s %(message)s',
                                    logger=self)
            else:
                self.disabled = True
        else:
            coloredlogs.install(fmt='%(asctime)s %(hostname)s %(levelname)s %(message)s',
                                logger=self)

        self.project = project
        if viz_backend is not None and not self.disabled:
            assert viz_backend in BACKENDS, "Please specify either None or a backend in {}".format(
                BACKENDS)
            self._create_backend(config, project, viz_backend, sync)
            self.run_id = self.backend.run_id
        else:
            self.backend = None
            self.run_id = "".join(random.SystemRandom().choice(string.ascii_lowercase +
                                                               string.digits)
                                  for _ in range(RUN_ID_LENGTH))
        
        # For writing to file purpose
        logging.basicConfig(filename=name+'.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    def _create_backend(self, config, project, backend, sync):
        if backend == 'wandb':
            self.backend = _WandBBackend(config, project, sync)
        elif backend == 'tensorboardx':
            self.backend = _TensorboardXBackend(config, project, sync)

    def update_config(self, config):
        """Save a configuration/set of parameters. This will overwrite any parameters that have
        already been saved.

        Parameters
        ----------
        config: Union[dict, OrderedDict]
            A dictionary containing configuration parameters.
        """
        if self.backend is not None:
            self.backend.update_config(config)

    def log_values(self, values, step=None):
        """Save a dictionary of values. If a step is specified, will save the values at that
        step. Otherwise, will increment to the next step.

        Parameters
        ----------
        config: Union[dict, OrderedDict]
            A dictionary containing values to log. Nested dictionaries i.e. {x: {y: 2}}
            will be flattened and saved as `x.y =2`. Only numerical types can be logged.

        step: int, default: None
            Corresponding timestamp for log. Once a step has been incremented, it's not possible
            to log at a previous time step.
        """
        if self.backend is not None:
            self.backend.log_values(values, step)

    def log_image(self, image, label, size=None, caption=None, step=None):
        """Save an image. If a step is specified, will save the image at that
        step. Otherwise, will increment to the next step.

        Parameters
        ----------
        image: Union[PIL.Image, numpy.ndarray, torch.Tensor]
            If PyTorch tensor, must be of shape [C, H, W]. C must be either 3, or 1. Does not
            support CUDA tensors, please place on CPU first.
            If numpy array, must be shape [H, W, C]. C must be either 3, or 1
            If PIL, must be RGB or L mode.

        label: str
            Group label for images.

        size: tuple, default: None
            Tuple of (H, W) for image to be logged. If None, will log default size.

        caption: str, default: None
            Caption for a particular image.

        step: int, default: None
            Corresponding timestamp for log. Once a step has been incremented, it's not possible
            to log at a previous time step.
        """
        if self.backend is not None:
            image = _prepare_image_for_logging(image)
            self.backend.log_image(image, label, size, caption, step)

    def log_plot(self, plot, label, caption=None, step=None):
        """Save a matplotlib plot. If a step is specified, will save the plot at that
        step. Otherwise, will increment to the next step.

        Parameters
        ----------
        plot: matplotlib.figure.Figure
            A matplotlib figure object

        label: str
            Group label for plots

        caption: str, default: None
            Caption for a particular plot

        step: int, default: None
            Corresponding timestamp for log. Once a step has been incremented, it's not possible
            to log at a previous time step.
        """
        if self.backend is not None:
            self.backend.log_plot(plot, label, caption, step)

    def end_log(self):
        """Finish logging and clean up."""
        if self.backend is not None:
            self.backend.end_log()


class _Backend:
    """Defines an API for visualization backends. Can be used to add more visualization tools,
    i.e. Visdom, etc.. if desired."""

    def __init__(self, config, project, run_id, log_dir, sync=True):
        """Base API for logging backends."""
        self.run_id = run_id
        self.log_dir = log_dir
        self.step = 0
        self.sync = sync
        self.config = config
        self.project = project

    def update_config(self, config):
        raise NotImplementedError

    def log_values(self, values, step=None):
        raise NotImplementedError

    def log_image(self, image, label, size=None, caption=None, step=None):
        raise NotImplementedError

    def log_plot(self, plot, label, caption=None, step=None):
        raise NotImplementedError

    def end_log(self):
        raise NotImplementedError

    def _increment_step(self, step):
        if step is not None:
            if step > self.step:
                self.step = step
        else:
            self.step += 1


class _WandBBackend(_Backend):
    """A WandB backend for visualization. Usage docs here:
    https://docs.wandb.com/docs/started.html
    """

    def __init__(self, config, project, sync=True):

        os.environ['WANDB_MODE'] = 'run' if sync else 'dryrun'
        wandb.init(config=config, project=project)
        super().__init__(config, project, wandb.run.id, wandb.run.dir, sync=sync)

    def update_config(self, config):
        self.config.update(config)
        wandb.config.update(config, allow_val_change=True)

    def log_values(self, values, step=None):
        values = collections.OrderedDict(values)
        values = _flatten_and_filter_dict(values, only_scalars=True)
        self._increment_step(step)
        wandb.log(values, step=self.step)

    def log_image(self, image, label, size=None, caption=None, step=None):
        """Save an image. If a step is specified, will save the image at that
        step. Otherwise, will increment to the next step.

        Parameters
        ----------
        image: Union[PIL.Image, numpy.ndarray, torch.Tensor]
            If PyTorch tensor, must be of shape [C, H, W]. C must be either 3, or 1. Does not
            support CUDA tensors, please place on CPU first.
            If numpy array, must be shape [H, W, C]. C must be either 3, or 1
            If PIL, must be RGB or L mode.

        label: str
            Group label for images.

        size: tuple, default: None
            Tuple of (H, W) for image to be logged. If None, will log default size.

        caption: str, default: None
            Caption for a particular image.

        step: int, default: None
            Corresponding timestamp for log. Once a step has been incremented, it's not possible
            to log at a previous time step.
        """
        if size is not None:
            image = np.array(
                Image.fromarray(image.astype('uint8')).resize((size[-1], size[0]),
                                                              resample=Image.BILINEAR))
        self._increment_step(step)
        wandb.log({label: [wandb.Image(image, caption=caption)]}, step=self.step)

    def log_plot(self, plot, label, caption=None, step=None):
        self._increment_step(step)
        wandb.log({label: plot}, step=self.step)

    def end_log(self):
        wandb.join()


class _TensorboardXBackend(_Backend):
    """A TensorboardX backend for visualization. Will create a directory called `tensorboardx`
    with a random hash assigned to the run in a similar manner to WandB. To view logs:

    `tensorboard --logdir tensorboardx/`

    Usage docs here: https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
    """

    def __init__(self, config, project, sync=False):
        HASH_LENGTH = 8
        self.start_time = time.time()
        self.start_asc_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        run_id = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits)
                         for _ in range(HASH_LENGTH))
        log_dir = '/'.join(['tensorboardx', project, 'run-' + self.start_asc_time + '-' + run_id])

        super().__init__(config, project, run_id, log_dir, sync=sync)
        self.logger = SummaryWriter(self.log_dir)
        self.update_config(config)

    def update_config(self, config):
        self.config.update(config)
        with open(self.log_dir + '/run-' + self.start_asc_time + '-' + self.run_id + '.json',
                  'w') as _cfg:
            json.dump(self.config, _cfg)

    def log_values(self, values, step=None):
        values = _flatten_and_filter_dict(values, only_scalars=True)
        self._increment_step(step)
        for _k, _v in values.items():
            self.logger.add_scalar(_k, _v, global_step=self.step)

    def log_image(self, image, label, size=None, caption=None, step=None):
        """Save an image. If a step is specified, will save the image at that
        step. Otherwise, will increment to the next step.

        Parameters
        ----------
        image: Union[PIL.Image, numpy.ndarray, torch.Tensor]
            If PyTorch tensor, must be of shape [C, H, W]. C must be either 3, or 1. Does not
            support CUDA tensors, please place on CPU first.
            If numpy array, must be shape [H, W, C]. C must be either 3, or 1
            If PIL, must be RGB or L mode.

        label: str
            Group label for images.

        size: tuple, default: None
            Tuple of (H, W) for image to be logged. If None, will log default size.

        caption: str, default: None
            Caption for a particular image.

        step: int, default: None
            Corresponding timestamp for log. Once a step has been incremented, it's not possible
            to log at a previous time step.
        """
        if size is not None:
            image = np.array(
                Image.fromarray(np.uint8(image)).resize((size[-1], size[0]),
                                                        resample=Image.BILINEAR))
        # Tranpose to (C, H, W) for TensorboardX
        image = image.transpose(2, 0, 1)
        self._increment_step(step)
        self.logger.add_image(label, image, global_step=self.step)

    def log_plot(self, plot, label, caption=None, step=None):
        self._increment_step(step)
        self.logger.add_figure(label, plot, global_step=self.step)

    def end_log(self):
        duration = time.time() - self.start_time
        self.update_config({'duration': duration})
        self.logger.close()


def _prepare_image_for_logging(image):
    """Converts torch.Tensor, PIL.Image, or np.array to to a 3 channel,
    [0,255] numpy.ndarray

    Returns
    -------
    image: numpy.ndarray
        np.uint8 image of shape W, H, 3
    """

    if isinstance(image, torch.Tensor):
        channels, _, _ = image.shape
        assert channels == 3 or channels == 1, "Expecting tensor of shape [C, H, W], C: 3 or 1"
        # image = image.transpose(2, 0).numpy()
        image = image.permute(1, 2, 0).numpy()

    elif isinstance(image, np.ndarray):
        _, _, channels = image.shape
        assert channels == 3 or channels == 1, "Expecting numpy.array of shape [H, W, C], C: 3 or 1"

    elif isinstance(image, Image.Image):
        image = np.array(image)
        _, _, channels = image.shape
    else:
        assert False, "Type {} of image not accepted".format(type(image))

    if channels == 1 or len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    # Normalize if not normalized
    if float(np.max(image)) > 255:
        image = image / float(np.max(image)) * 255

    return image.astype('uint8')


def _flatten_and_filter_dict(dictionary, parent_key='', sep='.', only_scalars=False):
    """Helper function that flattens nested dictionaries.

    Parameters
    ----------
    dictionary: Union[dict, OrderedDict]
        Dictionary to be flattened

    parent_key: str, default: ''
        Prefix to use for keys in flattened dictionary

    sep: str, default: '.'
        Separator to use when flattening keys from nested elements.
        e.g., by default:
        {dog: {cat: 0, mouse: 1}} -> {dog.cat: 0, dog.mouse: 1}

    only_scalars: bool, default: False
        If true, flattened dictionary only accept values that are scalars
    """
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(
                _flatten_and_filter_dict(v, new_key, sep=sep, only_scalars=only_scalars).items())
        else:
            if only_scalars:
                if isinstance(v, numbers.Number):
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
    return dict(items)
