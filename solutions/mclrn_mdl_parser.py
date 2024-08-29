import yaml
import os
from typedef import CTest, CRet, CModel
from typedef import TReturnClass, TReturnNames, TFactorName
# from typedef import TSigArgsSS, TSigArgsTSDB
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
            arg_val_combs = list(product(*model_args.values()))
            arg_combs = [{k: v for k, v in zip(model_args, vi)} for vi in arg_val_combs]
            for arg_comb in arg_combs:
                for sector in sectors:
                    sec_mdl_args = {
                        "model_type": model_type,
                        "model_args": arg_comb,
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


def get_tests(config_models: dict[str, dict]) -> list[CTest]:
    tests: list[CTest] = []
    for unique_id, m in config_models.items():
        ret = CRet(ret_class=m["ret_class"], ret_name=m["ret_name"], shift=m["shift"])
        model = CModel(model_type=m["model_type"], model_args=m["model_args"])
        test = CTest(unique_Id=unique_id, trn_win=m["trn_win"], sector=m["sector"], ret=ret, model=model)
        tests.append(test)
    return tests

# def get_signal_args_ss(tests: list[CTest], prediction_dir: str, signals_dir: str) -> list[TSigArgsSS]:
#     res: list[TSigArgsSS] = []
#     for test in tests:
#         input_dir = os.path.join(prediction_dir, test.save_tag_prd)
#         output_dir = os.path.join(signals_dir, test.save_tag_prd)
#         sid = test.ret.ret_name
#         sig_args = TSigArgsSS((input_dir, output_dir, sid))
#         res.append(sig_args)
#     return res
#
#
# def get_signal_args_tsdb(tests: list[CTest]) -> list[TSigArgsTSDB]:
#     res: list[TSigArgsTSDB] = []
#     for test in tests:
#         # factor=f"signals.001L1-NEU.W060.TR01.Ridge-A00.AGR.M0005"
#         # fields=["CloseRtn001L1-NEU"]
#         factor = TFactorName(".".join(["signals"] + test.prefix))
#         fields = TReturnNames([test.ret.ret_name])
#         sig_args = TSigArgsTSDB((factor, fields))
#         res.append(sig_args)
#     return res
