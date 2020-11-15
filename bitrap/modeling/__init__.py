__all__ = ['build_model']

from .bitrap_gmm import BiTraPGMM
from .bitrap_np import BiTraPNP

_MODELS_ = {
    'BiTraPNP': BiTraPNP,
    'BiTraPGMM': BiTraPGMM,
}

def make_model(cfg):
    model = _MODELS_[cfg.METHOD]
    try:
        return model(cfg, dataset_name=cfg.DATASET.NAME)
    except:
        return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)
