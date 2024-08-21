import os
import scipy.stats as sps
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track
from loguru import logger
from husfort.qutility import qtimer, SFG, error_handler, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from solutions.shared import gen_tst_ret_raw_db


class _CTstRetGeneric:
    def __init__(
            self,
            lag: int, win: int, universe: list[str],
            db_tst_ret_save_dir: str,
            db_struct_preprocess: CDbStruct
    ):
        self.lag = lag
        self.win = win
        self.universe = universe
        self.db_tst_ret_save_dir = db_tst_ret_save_dir
        self.db_struct_preprocess = db_struct_preprocess
        self.ret_lbl_cls = f"ClsRtn{self.save_id}"
        self.ret_lbl_opn = f"OpnRtn{self.save_id}"

    @property
    def tot_shift(self) -> int:
        return self.lag + self.win

    @property
    def rets(self) -> list[str]:
        return [self.ret_lbl_cls, self.ret_lbl_opn]

    @property
    def save_id(self) -> str:
        raise NotImplementedError

    def get_base_date(self, this_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(this_date, -self.tot_shift)

    def load_preprocess(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_preprocess.db_save_dir,
            db_name=f"{instru}.db",
            table=self.db_struct_preprocess.table,
            mode="r",
        )
        data = sqldb.read_by_range(
            bgn_date=bgn_date, stp_date=stp_date,
            value_columns=["trade_date", "ticker_major", "return_c_major", "return_o_major"]
        )
        return data


class CTstRetRaw(_CTstRetGeneric):
    @property
    def save_id(self) -> str:
        return f"{self.win:03d}L{self.lag}RAW"

    def cal_test_return(self, instru_ret_data: pd.DataFrame, base_bgn_date: str, base_end_date: str) -> pd.DataFrame:
        ret_cls, ret_opn = "return_c_major", "return_o_major"
        instru_ret_data[self.ret_lbl_cls] = instru_ret_data[ret_cls].rolling(window=self.win).sum().shift(
            -self.tot_shift)
        instru_ret_data[self.ret_lbl_opn] = instru_ret_data[ret_opn].rolling(window=self.win).sum().shift(
            -self.tot_shift)
        res = instru_ret_data.query(f"trade_date >= '{base_bgn_date}' & trade_date <= '{base_end_date}'")
        res = res[["trade_date", "ticker_major"] + self.rets]
        return res

    def process_for_instru(
            self,
            instru: str,
            bgn_date: str,
            stp_date: str,
            calendar: CCalendar,
    ):
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        base_bgn_date = self.get_base_date(iter_dates[0], calendar)
        base_end_date = self.get_base_date(iter_dates[-1], calendar)

        check_and_makedirs(dst_dir := os.path.join(self.db_tst_ret_save_dir, self.save_id))
        db_struct_instru = gen_tst_ret_raw_db(instru, dst_dir, self.rets)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="a",
        )
        if sqldb.check_continuity(base_bgn_date, calendar) == 0:
            instru_ret_data = self.load_preprocess(instru, base_bgn_date, stp_date)
            y_instru_data = self.cal_test_return(instru_ret_data, base_bgn_date, base_end_date)
            sqldb.update(update_data=y_instru_data)
        return 0

    @qtimer
    def main_test_return_raw(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        desc = f"Processing test return with lag = {SFG(self.lag)}, win = {SFG(self.win)}"
        for instru in track(self.universe, description=desc):
            self.process_for_instru(instru, bgn_date=bgn_date, stp_date=stp_date, calendar=calendar)
        return 0

# def trans_raw_to_rank(raw_data: pd.DataFrame, ref_cols: list[str]) -> pd.DataFrame:
#     target = raw_data[ref_cols]
#     res = target.rank() / (target.count() + 1)
#     return res
#
#
# class CTstRetNeu(CTstRetGeneric):
#     def __init__(
#             self,
#             lag: int,
#             win: int,
#             universe: list[str],
#             major_dir: str,
#             available_dir: str,
#             save_root_dir: str,
#     ):
#         super().__init__(lag, win, universe, save_root_dir=save_root_dir, major_dir=major_dir)
#         self.available_dir = available_dir
#         self.ref_id = f"{self.win:03d}L{self.lag}"
#         self.ref_dir = os.path.join(save_root_dir, self.ref_id)
#         self.ref_rets = [z.replace("-NEU", "") for z in self.rets]
#
#     @property
#     def save_id(self) -> str:
#         return f"{self.win:03d}L{self.lag}-NEU"
#
#     def get_tst_ret_data_type(self) -> np.dtype:
#         return np.dtype(
#             [
#                 ("tp", np.int64),
#                 ("trade_date", np.int32),
#                 ("ticker", "S8"),
#             ]
#             + [(_, np.float64) for _ in self.rets]
#         )
#
#     def load_ref_ret_by_instru(self, instru: str) -> pd.DataFrame:
#         ref_ret_by_instru_file = f"{instru}.ss"
#         ref_ret_by_instru_path = os.path.join(self.ref_dir, ref_ret_by_instru_file)
#         ss = PySharedStackPlus(pth=ref_ret_by_instru_path)
#         df = pd.DataFrame(ss.read_all())
#         return df
#
#     def load_ref_tst_ret(self, base_bgn_date: np.int32, base_end_date: np.int32) -> pd.DataFrame:
#         ref_dfs: list[pd.DataFrame] = []
#         for instru in self.universe:
#             df = self.load_ref_ret_by_instru(instru)
#             df = df.set_index("trade_date").truncate(before=int(base_bgn_date), after=int(base_end_date))
#             df["instrument"] = instru
#             ref_dfs.append(df)
#         res = pd.concat(ref_dfs, axis=0, ignore_index=False)
#         res = res.reset_index().sort_values(by=["trade_date"], ascending=True)
#         res = res[["tp", "trade_date", "instrument"] + self.ref_rets]
#         return res
#
#     def load_available(self, base_bgn_date: np.int32, base_end_date: np.int32) -> pd.DataFrame:
#         available_file = "available.ss"
#         available_path = os.path.join(self.available_dir, available_file)
#         ss = PySharedStackPlus(available_path)
#         df = pd.DataFrame(ss.read_all())
#         df = df.set_index("trade_date").truncate(before=int(base_bgn_date), after=int(base_end_date)).reset_index()
#         df["instrument"] = df["instrument"].map(lambda z: z.decode("utf-8"))
#         df = df[["tp", "trade_date", "instrument", "sectorL1"]]
#         return df
#
#     def normalize(self, data: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
#         pool, jobs = mp.Pool(processes=48), []
#         for _, sub_df in data.groupby(group_keys):
#             job = pool.apply_async(trans_raw_to_rank, args=(sub_df, self.ref_rets), error_callback=error_handler)
#             jobs.append(job)
#         pool.close()
#         pool.join()
#         dfs: list[pd.DataFrame] = [job.get() for job in jobs]
#         rank_df = pd.concat(dfs, axis=0, ignore_index=False)
#         normalize_df = pd.DataFrame(
#             data=sps.norm.ppf(rank_df.values),
#             index=rank_df.index,
#             columns=self.rets,
#         )
#         return normalize_df
#
#     def neutralize_by_date(self, net_ref_tst_ret_data: pd.DataFrame) -> pd.DataFrame:
#         normalize_df = self.normalize(net_ref_tst_ret_data, group_keys=["trade_date", "sectorL1"])
#         if (s0 := len(normalize_df)) != (s1 := len(net_ref_tst_ret_data)):
#             raise ValueError(f"[ERR] Size after normalization = {s0} != Size before normalization {s1}")
#         else:
#             merge_df = net_ref_tst_ret_data[["tp", "trade_date", "instrument"]].merge(
#                 right=normalize_df[self.rets],
#                 left_index=True,
#                 right_index=True,
#                 how="left",
#             )
#             res = merge_df[["tp", "trade_date", "instrument"] + self.rets]
#             return res
#
#     def load_header(self, instru: str, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
#         df = self.load_preprocess(instru)
#         df = df.set_index("trade_date").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
#         return df[["tp", "trade_date", "ticker"]]
#
#     def process_by_instru(
#             self,
#             instru: str,
#             instru_tst_ret_neu_data: pd.DataFrame,
#             base_bgn_date: np.int32,
#             base_end_date: np.int32,
#             calendar,
#     ):
#         instru_header = self.load_header(instru, base_bgn_date, base_end_date)
#         instru_data = pd.merge(
#             left=instru_header,
#             right=instru_tst_ret_neu_data,
#             on=["tp", "trade_date"],
#             how="left",
#         )
#         tst_ret_data_type = self.get_tst_ret_data_type()
#         instru_tst_ret_neu_data = instru_data[["tp", "trade_date", "ticker"] + self.rets]
#         self.save(instru, instru_tst_ret_neu_data, tst_ret_data_type=tst_ret_data_type, calendar=calendar)
#         return 0
#
#     @qtimer
#     def main_test_return_neutralize(
#             self,
#             bgn_date: np.int32,
#             end_date: np.int32,
#             calendar: CCalendar,
#             call_multiprocess: bool,
#             processes: int,
#     ):
#         print(f"[INF] Neutralizing test return with lag = {SFG(self.lag)}, win = {SFG(self.win)}")
#         base_bgn_date = self.get_base_date(bgn_date, calendar)
#         base_end_date = self.get_base_date(end_date, calendar)
#         ref_tst_ret_data = self.load_ref_tst_ret(base_bgn_date, base_end_date)
#         available_data = self.load_available(base_bgn_date, base_end_date)
#         net_ref_tst_ret_data = pd.merge(
#             left=available_data,
#             right=ref_tst_ret_data,
#             on=["tp", "trade_date", "instrument"],
#             how="left",
#         ).sort_values(by=["trade_date", "sectorL1"])
#         tst_ret_neu_data = self.neutralize_by_date(net_ref_tst_ret_data)
#         if call_multiprocess:
#             with mp.get_context("spawn").Pool(processes=processes) as pool:
#                 for instru, instru_tst_ret_neu_data in tst_ret_neu_data.groupby(by="instrument"):
#                     pool.apply_async(
#                         self.process_by_instru,
#                         args=(instru, instru_tst_ret_neu_data, base_bgn_date, base_end_date, calendar),
#                         error_callback=error_handler,
#                     )
#                 pool.close()
#                 pool.join()
#         else:
#             for instru, instru_tst_ret_neu_data in tst_ret_neu_data.groupby(by="instrument"):
#                 self.process_by_instru(
#                     instru, instru_tst_ret_neu_data, base_bgn_date, base_end_date, calendar  # type:ignore
#                 )
#         return 0
