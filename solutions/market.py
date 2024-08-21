import numpy as np
import pandas as pd
from loguru import logger
from husfort.qutility import qtimer, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from solutions.shared import convert_mkt_idx


def load_available(db_struct: CDbStruct, bgn_date: str, stp_date: str) -> pd.DataFrame:
    sqldb = CMgrSqlDb(
        db_save_dir=db_struct.db_save_dir,
        db_name=db_struct.db_name,
        table=db_struct.table,
        mode="r"
    )
    avlb_data = sqldb.read_by_range(bgn_date=bgn_date, stp_date=stp_date)
    return avlb_data


def cal_market_return_by_date(sub_data: pd.DataFrame) -> float:
    wgt = sub_data["rel_wgt"] / sub_data["rel_wgt"].sum()
    ret = sub_data["return"] @ wgt
    return ret


def cal_market_return(
        bgn_date: str,
        stp_date: str,
        db_struct_avlb: CDbStruct,
        sectors: list[str]
) -> pd.DataFrame:
    available_data = load_available(db_struct=db_struct_avlb, bgn_date=bgn_date, stp_date=stp_date)
    input_for_return = available_data.set_index("trade_date")
    input_for_return["rel_wgt"] = np.sqrt(input_for_return["amount"])
    ret = {"market": input_for_return.groupby(by="trade_date").apply(cal_market_return_by_date)}
    for sector, sector_df in input_for_return.groupby(by="sectorL0"):
        ret[sector] = sector_df.groupby(by="trade_date").apply(cal_market_return_by_date)
    for sector, sector_df in input_for_return.groupby(by="sectorL1"):
        ret[sector] = sector_df.groupby(by="trade_date").apply(cal_market_return_by_date)
    ret_by_sector = pd.DataFrame(ret).reset_index()
    # --- reformat
    mkt_cols = ["market"]
    sec0_cols, sec1_cols = ["C"], sectors
    ret_by_sector = ret_by_sector[["trade_date"] + mkt_cols + sec0_cols + sec1_cols]
    return ret_by_sector


def load_market_index(bgn_date: str, stp_date: str, path_mkt_idx_data: str, mkt_idxes: list[str]) -> pd.DataFrame:
    mkt_idx_data = {}
    for mkt_idx in mkt_idxes:
        df = pd.read_excel(path_mkt_idx_data, sheet_name=mkt_idx, header=1)
        df["trade_date"] = df["Date"].map(lambda _: _.strftime("%Y%m%d"))
        mkt_idx_data[convert_mkt_idx(mkt_idx)] = df.set_index("trade_date")["pct_chg"] / 100
    mkt_idx_df = pd.DataFrame(mkt_idx_data).reset_index()
    mkt_idx_df = mkt_idx_df.query(expr=f"trade_date >= '{bgn_date}' & trade_date < '{stp_date}'")
    return mkt_idx_df


def merge_mkt_idx(ret_by_sector: pd.DataFrame, mkt_idx_df: pd.DataFrame) -> pd.DataFrame:
    if (s0 := len(ret_by_sector)) != (s1 := len(mkt_idx_df)):
        logger.info(f"length of custom market index = {s0}")
        logger.info(f"length of        market index = {s1}")
        d0 = set(ret_by_sector["trade_date"])
        d1 = set(mkt_idx_df["trade_date"])
        in_d0_not_in_d1 = d0 - d1
        in_d1_not_in_d0 = d1 - d0
        if in_d0_not_in_d1:
            logger.info(f"the following days are in custom but not in official {in_d0_not_in_d1}")
        if in_d1_not_in_d0:
            logger.info(f"the following days are in official but not in custom {in_d1_not_in_d0}")
    new_data = pd.merge(left=ret_by_sector, right=mkt_idx_df, on="trade_date", how="right")
    return new_data


def sort_columns(new_data: pd.DataFrame, db_struct_mkt: CDbStruct) -> pd.DataFrame:
    return new_data[db_struct_mkt.table.vars.names]


@qtimer
def main_market(
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        db_struct_avlb: CDbStruct,
        db_struct_mkt: CDbStruct,
        path_mkt_idx_data: str,
        mkt_idxes: list[str],
        sectors: list[str],
):
    check_and_makedirs(db_struct_mkt.db_save_dir)
    sqldb = CMgrSqlDb(
        db_save_dir=db_struct_mkt.db_save_dir,
        db_name=db_struct_mkt.db_name,
        table=db_struct_mkt.table,
        mode="a",
    )

    if sqldb.check_continuity(bgn_date, calendar) == 0:
        ret_by_sector = cal_market_return(bgn_date, stp_date, db_struct_avlb, sectors=sectors)
        mkt_idx_df = load_market_index(bgn_date, stp_date, path_mkt_idx_data, mkt_idxes)
        new_data = merge_mkt_idx(ret_by_sector, mkt_idx_df)
        new_data = sort_columns(new_data, db_struct_mkt)
        print(new_data)
        sqldb.update(update_data=new_data)
    return 0
