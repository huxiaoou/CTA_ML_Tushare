from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar


def convert_mkt_idx(mkt_idx: str, prefix: str = "I") -> str:
    return f"{prefix}{mkt_idx.replace('.', '_')}"


def gen_tst_ret_raw_db(instru: str, db_save_dir: str, rets: list[str]) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_dir,
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="test_return_raw",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(ret, "REAL") for ret in rets]
        )
    )
