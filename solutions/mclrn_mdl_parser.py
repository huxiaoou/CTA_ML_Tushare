import yaml
import os
from typedef import TReturnClass, TReturnNames
from itertools import product
from husfort.qutility import check_and_makedirs


def parse_model_configs(
        models: dict,
        ret_class: TReturnClass,
        ret_names: TReturnNames,
        shift: int,
        sectors: list[str],
        trn_wins: list[int],
        cfg_mdl_dir: str,
        cfg_mdl_file: str,
):
    path_config_models = os.path.join(cfg_mdl_dir, cfg_mdl_file)
    m, iter_args = 0, {}
    for ret_name, trn_win in product(ret_names, trn_wins):
        shared_args = {
            "ret_class": ret_class,
            "ret_name": ret_name,
            "shift": shift,
            "trn_win": trn_win,
        }
        for model_type, model_args in models.items():
            for sector in sectors:
                sec_mdl_args = {
                    "model_type": model_type,
                    "model_args": model_args,
                    "sector": sector,
                }
                iter_args[f"M{m:04d}"] = {**shared_args, **sec_mdl_args}
                m += 1
    check_and_makedirs(cfg_mdl_dir)
    with open(path_config_models, "w+") as f:
        yaml.dump_all([iter_args], f)
    return 0


def load_config_models(cfg_mdl_dir: str, cfg_mdl_file: str) -> dict[str, dict]:
    model_config_path = os.path.join(cfg_mdl_dir, cfg_mdl_file)
    with open(model_config_path, "r") as f:
        config_models = yaml.safe_load(f)
    return config_models
