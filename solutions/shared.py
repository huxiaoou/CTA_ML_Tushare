import os
from husfort.qutility import SFR
from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar
from typedef import TFactorClass, TFactorNames, CTestFtSlc, CTestMdl, CSimArgs
from typedef import CRet, CModel
from typedef import TSimArgsGrp, TSimArgsPriKey, TSimArgsSecKey
from typedef import TSimArgsGrpBySec, TSimArgsPriKeyBySec, TSimArgsSecKeyBySec
from typedef import CPortfolioArgs, TUniqueId


def convert_mkt_idx(mkt_idx: str, prefix: str = "I") -> str:
    return f"{prefix}{mkt_idx.replace('.', '_')}"


# ----------------------------------------
# ------ sqlite3 database structure ------
# ----------------------------------------

def gen_tst_ret_db(instru: str, db_save_root_dir: str, save_id: str, rets: list[str]) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, save_id),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="test_return",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(ret, "REAL") for ret in rets],
        )
    )


def gen_tst_ret_regrp_db(db_save_root_dir: str, ret_name: str) -> CDbStruct:
    """

    :param db_save_root_dir:
    :param ret_name: like "ClsRtn001L1RAW", "OpnRtn001L1NEU"
    :return:
    """

    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{ret_name}.db",
        table=CSqlTable(
            name="test_return",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(ret_name, "REAL")],
        )
    )


def gen_fac_db(instru: str, db_save_root_dir: str, factor_class: TFactorClass, factor_names: TFactorNames) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, factor_class),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(fn, "REAL") for fn in factor_names],
        )
    )


def gen_feat_slc_db(test: CTestFtSlc, db_save_root_dir: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, test.save_id),
        db_name=f"{test.sector}.db",
        table=CSqlTable(
            name="feature_selection",
            primary_keys=[
                CSqlVar("trade_date", "TEXT"),
                CSqlVar("factor_class", "TEXT"),
                CSqlVar("factor_name", "TEXT"),
            ],
            value_columns=[CSqlVar("is_neu", "INTEGER")],
        )
    )


def gen_prdct_db(db_save_root_dir: str, test: CTestMdl) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{test.save_tag_mdl}.db",
        table=CSqlTable(
            name="prediction",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(test.ret.ret_name, "REAL")],
        )
    )


def gen_sig_mdl_db(db_save_root_dir: str, test: CTestMdl) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{test.save_tag_mdl}.db",
        table=CSqlTable(
            name="signals",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar("weight", "REAL")],
        )
    )


def gen_sig_pfo_db(db_save_root_dir: str, portfolio_id: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{portfolio_id}.db",
        table=CSqlTable(
            name="signals",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar("weight", "REAL")],
        )
    )


def gen_nav_db(db_save_dir: str, save_id: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name="nav",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[
                CSqlVar("raw_ret", "REAL"),
                CSqlVar("dlt_wgt", "REAL"),
                CSqlVar("cost", "REAL"),
                CSqlVar("net_ret", "REAL"),
                CSqlVar("nav", "REAL"),
            ],
        )
    )


# -----------------------------------------
# ------ arguments about simulations ------
# -----------------------------------------

def gen_model_tests(config_models: dict[str, dict]) -> list[CTestMdl]:
    tests: list[CTestMdl] = []
    for unique_id, m in config_models.items():
        ret = CRet(ret_class=m["ret_class"], ret_name=m["ret_name"], shift=m["shift"])
        model = CModel(model_type=m["model_type"], model_args=m["model_args"])
        test = CTestMdl(unique_Id=unique_id, trn_win=m["trn_win"], sector=m["sector"], ret=ret, model=model)
        tests.append(test)
    return tests


def get_sim_args_from_test_models(
        test_mdls: list[CTestMdl], cost: float, test_return_dir: str, signals_mdl_dir: str
) -> list[CSimArgs]:
    res: list[CSimArgs] = []
    for test_mdl in test_mdls:
        if test_mdl.ret.ret_name.startswith("Opn"):
            tgt_ret_class, tgt_ret_name = "001L1", "OpnRtn001L1RAW"
        elif test_mdl.ret.ret_name.startswith("Cls"):
            tgt_ret_class, tgt_ret_name = "001L1", "ClsRtn001L1RAW"
        else:
            raise ValueError(f"ret_name = {SFR(test_mdl.ret.ret_name)} is illegal")
        tgt_ret = CRet(tgt_ret_class, tgt_ret_name, 2)

        db_struct_ret = gen_tst_ret_regrp_db(db_save_root_dir=test_return_dir, ret_name=tgt_ret_name)
        db_struct_sig = gen_sig_mdl_db(db_save_root_dir=signals_mdl_dir, test=test_mdl)
        sim_id = f"{test_mdl.save_tag_mdl}.T{tgt_ret_name}"
        sim_args = CSimArgs(
            sim_id=sim_id,
            tgt_ret=tgt_ret,
            db_struct_sig=db_struct_sig,
            db_struct_ret=db_struct_ret,
            cost=cost,
        )
        res.append(sim_args)
    return res


def get_sim_args_from_portfolios(
        portfolios: dict[TUniqueId, dict], cost: float, test_return_dir: str, signals_pfo_dir: str
) -> list[CSimArgs]:
    res: list[CSimArgs] = []
    for portfolio_id, portfolio_cfg in portfolios.items():
        target: str = portfolio_cfg["target"]
        if target.startswith("Opn"):
            tgt_ret_class, tgt_ret_name = "001L1", "OpnRtn001L1RAW"
        elif target.startswith("Cls"):
            tgt_ret_class, tgt_ret_name = "001L1", "ClsRtn001L1RAW"
        else:
            raise ValueError(f"target = {target} is illegal")
        tgt_ret = CRet(tgt_ret_class, tgt_ret_name, 2)

        db_struct_ret = gen_tst_ret_regrp_db(db_save_root_dir=test_return_dir, ret_name=tgt_ret_name)
        db_struct_sig = gen_sig_pfo_db(db_save_root_dir=signals_pfo_dir, portfolio_id=portfolio_id)
        sim_args = CSimArgs(
            sim_id=portfolio_id,
            tgt_ret=tgt_ret,
            db_struct_sig=db_struct_sig,
            db_struct_ret=db_struct_ret,
            cost=cost,
        )
        res.append(sim_args)
    return res


def group_sim_args(sim_args_list: list[CSimArgs]) -> TSimArgsGrp:
    grouped_sim_args: TSimArgsGrp = {}
    for sim_args in sim_args_list:
        ret_class, trn_win, model_desc, sector, unique_id, ret_name, tgt_ret_name = sim_args.sim_id.split(".")
        key0, key1 = TSimArgsPriKey((ret_class, trn_win, model_desc, ret_name)), TSimArgsSecKey((sector, unique_id))
        if key0 not in grouped_sim_args:
            grouped_sim_args[key0] = {}
        grouped_sim_args[key0][key1] = sim_args
    return grouped_sim_args


def group_sim_args_by_sector(sim_args_list: list[CSimArgs]) -> TSimArgsGrpBySec:
    grouped_sim_args: TSimArgsGrpBySec = {}
    for sim_args in sim_args_list:
        ret_class, trn_win, model_desc, sector, unique_id, ret_name, tgt_ret_name = sim_args.sim_id.split(".")
        key0, key1 = TSimArgsPriKeyBySec(sector), TSimArgsSecKeyBySec(
            (ret_class, trn_win, model_desc, ret_name, unique_id))
        if key0 not in grouped_sim_args:
            grouped_sim_args[key0] = {}
        grouped_sim_args[key0][key1] = sim_args
    return grouped_sim_args


def get_portfolio_args(portfolios: dict[TUniqueId, dict], sim_args_list: list[CSimArgs]) -> list[CPortfolioArgs]:
    res: list[CPortfolioArgs] = []
    for portfolio_id, portfolio_cfg in portfolios.items():
        target, weights = portfolio_cfg["target"], portfolio_cfg["weights"]
        portfolio_sim_args: dict[TUniqueId, CSimArgs] = {}
        for sim_args in sim_args_list:
            *_, unique_id, _, tgt_ret_name = sim_args.sim_id.split(".")
            if (unique_id in weights) and (tgt_ret_name[1:] == target):
                portfolio_sim_args[unique_id] = sim_args
        portfolio_arg = CPortfolioArgs(portfolio_id, target, weights, portfolio_sim_args)
        res.append(portfolio_arg)
    return res
