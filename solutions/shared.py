import os
from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar


def convert_mkt_idx(mkt_idx: str, prefix: str = "I") -> str:
    return f"{prefix}{mkt_idx.replace('.', '_')}"


def gen_tst_ret_db(instru: str, db_save_root_dir: str, save_id: str, rets: list[str]) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, save_id),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="test_return",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(ret, "REAL") for ret in rets]
        )
    )
