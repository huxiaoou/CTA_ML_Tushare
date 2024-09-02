import scipy.stats as sps
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track
from loguru import logger
from husfort.qutility import qtimer, SFG, error_handler, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from solutions.shared import gen_tst_ret_db, gen_tst_ret_regrp_db


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

        db_struct_instru = gen_tst_ret_db(
            instru=instru,
            db_save_root_dir=self.db_tst_ret_save_dir,
            save_id=self.save_id,
            rets=self.rets,
        )
        check_and_makedirs(db_struct_instru.db_save_dir)
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


class CTstRetNeu(_CTstRetGeneric):
    def __init__(
            self,
            lag: int,
            win: int,
            universe: list[str],
            db_tst_ret_save_dir: str,
            db_struct_preprocess: CDbStruct,
            db_struct_avlb: CDbStruct,
    ):
        super().__init__(lag, win, universe, db_tst_ret_save_dir, db_struct_preprocess)
        self.db_struct_avlb = db_struct_avlb
        self.ref_id = self.save_id.replace("NEU", "RAW")
        self.ref_rets = [z.replace("NEU", "RAW") for z in self.rets]

    @property
    def save_id(self) -> str:
        return f"{self.win:03d}L{self.lag}NEU"

    def load_ref_ret_by_instru(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_ref = gen_tst_ret_db(
            instru=instru,
            db_save_root_dir=self.db_tst_ret_save_dir,
            save_id=self.ref_id,
            rets=self.ref_rets,
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_ref.db_save_dir,
            db_name=db_struct_ref.db_name,
            table=db_struct_ref.table,
            mode="r"
        )
        ref_data = sqldb.read_by_range(bgn_date, stp_date)
        return ref_data

    def load_ref_ret(self, base_bgn_date: str, base_stp_date: str) -> pd.DataFrame:
        ref_dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            df = self.load_ref_ret_by_instru(instru, bgn_date=base_bgn_date, stp_date=base_stp_date)
            df["instrument"] = instru
            ref_dfs.append(df)
        res = pd.concat(ref_dfs, axis=0, ignore_index=False)
        res = res.reset_index().sort_values(by=["trade_date"], ascending=True)
        res = res[["trade_date", "instrument"] + self.ref_rets]
        return res

    def load_available(self, base_bgn_date: str, base_stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_avlb.db_save_dir,
            db_name=self.db_struct_avlb.db_name,
            table=self.db_struct_avlb.table,
            mode="r",
        )
        avlb_data = sqldb.read_by_range(bgn_date=base_bgn_date, stp_date=base_stp_date)
        avlb_data = avlb_data[["trade_date", "instrument", "sectorL1"]]
        return avlb_data

    @staticmethod
    def trans_raw_to_rank(raw_data: pd.DataFrame, ref_cols: list[str]) -> pd.DataFrame:
        target = raw_data[ref_cols]
        res = target.rank() / (target.count() + 1)
        return res

    def normalize(self, data: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
        pool, jobs = mp.Pool(), []
        for _, sub_df in data.groupby(group_keys):
            job = pool.apply_async(self.trans_raw_to_rank, args=(sub_df, self.ref_rets), error_callback=error_handler)
            jobs.append(job)
        pool.close()
        pool.join()
        dfs: list[pd.DataFrame] = [job.get() for job in jobs]
        rank_df = pd.concat(dfs, axis=0, ignore_index=False)
        normalize_df = pd.DataFrame(
            data=sps.norm.ppf(rank_df.values),
            index=rank_df.index,
            columns=self.rets,
        )
        return normalize_df

    def neutralize_by_date(self, net_ref_tst_ret_data: pd.DataFrame) -> pd.DataFrame:
        normalize_df = self.normalize(net_ref_tst_ret_data, group_keys=["trade_date", "sectorL1"])
        if (s0 := len(normalize_df)) != (s1 := len(net_ref_tst_ret_data)):
            raise ValueError(f"[ERR] Size after normalization = {s0} != Size before normalization {s1}")
        else:
            merge_df = net_ref_tst_ret_data[["trade_date", "instrument"]].merge(
                right=normalize_df[self.rets],
                left_index=True,
                right_index=True,
                how="left",
            )
            res = merge_df[["trade_date", "instrument"] + self.rets]
            return res

    def load_header(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        df = self.load_preprocess(instru, bgn_date, stp_date)
        df = df.rename(columns={"ticker_major": "ticker"})
        return df[["trade_date", "ticker"]]

    def process_by_instru(
            self,
            instru: str,
            instru_tst_ret_neu_data: pd.DataFrame,
            base_bgn_date: str,
            base_stp_date: str,
            calendar,
    ):
        db_struct_instru = gen_tst_ret_db(
            instru=instru,
            db_save_root_dir=self.db_tst_ret_save_dir,
            save_id=self.save_id,
            rets=self.rets,
        )
        check_and_makedirs(db_struct_instru.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="a",
        )
        if sqldb.check_continuity(base_bgn_date, calendar) == 0:
            instru_header = self.load_header(instru, bgn_date=base_bgn_date, stp_date=base_stp_date)
            instru_data = pd.merge(
                left=instru_header,
                right=instru_tst_ret_neu_data,
                on="trade_date",
                how="left",
            )
            instru_tst_ret_neu_data = instru_data[db_struct_instru.table.vars.names]
            sqldb.update(update_data=instru_tst_ret_neu_data)
        return 0

    @qtimer
    def main_test_return_neu(
            self,
            bgn_date: str,
            stp_date: str,
            calendar: CCalendar,
            call_multiprocess: bool,
            processes: int,
    ):
        logger.info(f"Neutralizing test return with lag = {SFG(self.lag)}, win = {SFG(self.win)}")
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        base_bgn_date = self.get_base_date(iter_dates[0], calendar)
        base_end_date = self.get_base_date(iter_dates[-1], calendar)
        base_stp_date = calendar.get_next_date(base_end_date, shift=1)

        ref_tst_ret_data = self.load_ref_ret(base_bgn_date, base_stp_date)
        available_data = self.load_available(base_bgn_date, base_stp_date)
        net_ref_tst_ret_data = pd.merge(
            left=available_data,
            right=ref_tst_ret_data,
            on=["trade_date", "instrument"],
            how="left",
        ).sort_values(by=["trade_date", "sectorL1"])
        tst_ret_neu_data = self.neutralize_by_date(net_ref_tst_ret_data)
        if call_multiprocess:
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for instru, instru_tst_ret_neu_data in tst_ret_neu_data.groupby(by="instrument"):
                    pool.apply_async(
                        self.process_by_instru,
                        args=(instru, instru_tst_ret_neu_data, base_bgn_date, base_stp_date, calendar),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
        else:
            for instru, instru_tst_ret_neu_data in tst_ret_neu_data.groupby(by="instrument"):
                self.process_by_instru(
                    instru=instru,  # type:ignore
                    instru_tst_ret_neu_data=instru_tst_ret_neu_data,
                    base_bgn_date=base_bgn_date,
                    base_stp_date=base_stp_date,
                    calendar=calendar,
                )
        return 0


def main_regroup(
        universe: list[str],
        win: int,
        lag: int,
        db_save_root_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        ret_types: tuple = ("RAW", "NEU"),
):
    iter_dates = calendar.get_iter_list(bgn_date, stp_date)
    base_bgn_date = calendar.get_next_date(iter_dates[0], shift=-(lag + win))
    base_end_date = calendar.get_next_date(iter_dates[-1], shift=-(lag + win))
    base_stp_date = calendar.get_next_date(base_end_date, shift=1)

    for ret_type in ret_types:
        save_id = f"{win:03d}L{lag}{ret_type}"
        ret_cls, ret_opn = f"ClsRtn{save_id}", f"OpnRtn{save_id}"
        rets = [ret_cls, ret_opn]

        # --- load
        dfs: list[pd.DataFrame] = []
        for instrument in universe:
            db_struct_instru = gen_tst_ret_db(
                instru=instrument,
                db_save_root_dir=db_save_root_dir,
                save_id=save_id,
                rets=rets,
            )
            sql_db = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            instru_data = sql_db.read_by_range(
                bgn_date=base_bgn_date, stp_date=base_stp_date, value_columns=["trade_date"] + rets
            )
            instru_data["instrument"] = instrument
            instru_data[rets] = instru_data[rets].astype(np.float64).fillna(np.nan)
            dfs.append(instru_data)
        all_ret_data = pd.concat(dfs, axis=0, ignore_index=True)
        all_ret_data = all_ret_data.sort_values(by=["trade_date", "instrument"])

        # save
        for ret_name in rets:
            ret_data = all_ret_data[["trade_date", "instrument", ret_name]]
            db_struct_ret = gen_tst_ret_regrp_db(
                db_save_root_dir=db_save_root_dir,
                ret_name=ret_name,
            )
            sql_db = CMgrSqlDb(
                db_save_dir=db_struct_ret.db_save_dir,
                db_name=db_struct_ret.db_name,
                table=db_struct_ret.table,
                mode="a",
            )
            if sql_db.check_continuity(ret_data["trade_date"].iloc[0], calendar) == 0:
                sql_db.update(ret_data)
            logger.info(f"Regrouping test return for {SFG(ret_name)}")
    return 0
