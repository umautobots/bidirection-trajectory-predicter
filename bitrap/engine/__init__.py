from bitrap.engine.trainer import do_train 
from bitrap.engine.trainer import do_val 
from bitrap.engine.trainer import inference

ENGINE_ZOO = {
                'BiTraPNP': (do_train, do_val, inference),
                'BiTraPGMM': (do_train, do_val, inference),
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]
