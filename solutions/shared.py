import os
from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar
from typedef import TFactorClass, TFactorNames, CTestFtSlc, CTest


def convert_mkt_idx(mkt_idx: str, prefix: str = "I") -> str:
    return f"{prefix}{mkt_idx.replace('.', '_')}"


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


def gen_prdct_db(db_save_root_dir: str, test: CTest) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{test.save_tag_mdl}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(test.ret.ret_name, "REAL")],
        )
    )


def gen_sig_mdl_db(db_save_root_dir: str, test: CTest) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{test.save_tag_mdl}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(test.ret.ret_name, "REAL")],
        )
    )
