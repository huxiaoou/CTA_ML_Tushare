import multiprocessing as mp
import numpy as np
import pandas as pd
from loguru import logger
from itertools import product
from rich.progress import track, Progress
from sklearn.feature_selection import mutual_info_regression
from husfort.qutility import qtimer, SFG, SFY, error_handler, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from typedef import TUniverse, TReturnName
from typedef import TFactorClass, TFactorName, TFactorNames, TFactorComb, TFactorsPool
from typedef import CTestFtSlc, CRet
from solutions.shared import gen_fac_db, gen_tst_ret_db, gen_feat_slc_db


class CFeatSlcReaderAndWriter:
    def __init__(self, test: CTestFtSlc, feat_slc_save_root_dir: str, tst_ret_save_root_dir: str):
        self.test = test
        self.db_struct_feat_slc = gen_feat_slc_db(test, feat_slc_save_root_dir)
        self.tst_ret_save_root_dir = tst_ret_save_root_dir

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_feat_slc.db_save_dir,
            db_name=self.db_struct_feat_slc.db_name,
            table=self.db_struct_feat_slc.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date).set_index("trade_date")
        return data

    def load_by_date(self, trade_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_feat_slc.db_save_dir,
            db_name=self.db_struct_feat_slc.db_name,
            table=self.db_struct_feat_slc.table,
            mode="r",
        )
        data = sqldb.read_by_date(trade_date)
        return data

    def save(self, new_data: pd.DataFrame):
        check_and_makedirs(self.db_struct_feat_slc.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_feat_slc.db_save_dir,
            db_name=self.db_struct_feat_slc.db_name,
            table=self.db_struct_feat_slc.table,
            mode="a",
        )
        sqldb.update(update_data=new_data)
        return 0

    def get_slc_facs_pool(
            self,
            trade_date: str,
            factors_by_instru_dir: str,
            neutral_by_instru_dir: str
    ) -> TFactorsPool:
        trade_date_data = self.load_by_date(trade_date)
        pool: TFactorsPool = []
        for (factor_class, is_neu), factor_class_data in trade_date_data.groupby(by=["factor_class", "is_neu"]):
            factor_names = factor_class_data["factor_name"].tolist()
            if is_neu == 0:
                comb = TFactorComb((factor_class, factor_names, factors_by_instru_dir))
            elif is_neu == 1:
                comb = TFactorComb((factor_class, factor_names, neutral_by_instru_dir))
            else:
                raise ValueError(f"is_neu = {is_neu} is illegal")
            pool.append(comb)
        return pool


class CFeatSlc(CFeatSlcReaderAndWriter):
    XY_INDEX = ["trade_date", "instrument"]
    RANDOM_STATE = 0

    def __init__(
            self,
            test: CTestFtSlc,
            feat_slc_save_root_dir: str,
            tst_ret_save_root_dir: str,
            db_struct_avlb: CDbStruct,
            universe: TUniverse,
            facs_pool: TFactorsPool,
    ):
        super().__init__(test, feat_slc_save_root_dir, tst_ret_save_root_dir)
        self.db_struct_avlb = db_struct_avlb
        self.universe = universe
        self.facs_pool = facs_pool
        self.mapper_name_to_class: dict[TFactorName, TFactorClass] = {}
        for factor_class, factor_names, _ in self.facs_pool:
            self.mapper_name_to_class.update({n: factor_class for n in factor_names})

    @property
    def x_cols(self) -> TFactorNames:
        ns = []
        for _, n, _ in self.facs_pool:
            ns.extend(n)
        return ns

    @property
    def y_col(self) -> TReturnName:
        return self.test.ret.ret_name

    @staticmethod
    def load_factor_by_instru(instru: str, factor_comb: TFactorComb, bgn_date: str, stp_date: str) -> pd.DataFrame:
        factor_class, factor_names, db_save_root_dir = factor_comb
        db_struct_fac = gen_fac_db(instru, db_save_root_dir, factor_class, factor_names)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_fac.db_save_dir,
            db_name=db_struct_fac.db_name,
            table=db_struct_fac.table,
            mode="r",
        )
        instru_data = sqldb.read_by_range(bgn_date, stp_date, value_columns=["trade_date"] + factor_names)
        instru_data[factor_names] = instru_data[factor_names].astype(np.float64).fillna(np.nan)
        return instru_data

    def load_factor(self, bgn_date: str, stp_date: str, factor_comb: TFactorComb) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            instru_data = self.load_factor_by_instru(
                instru=instru, factor_comb=factor_comb,
                bgn_date=bgn_date, stp_date=stp_date,
            )
            instru_data["instrument"] = instru
            dfs.append(instru_data)
        factor_data = pd.concat(dfs, axis=0, ignore_index=True)
        factor_data = factor_data.set_index(self.XY_INDEX).sort_index()
        return factor_data

    def load_x(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        factor_dfs: list[pd.DataFrame] = []
        for factor_comb in self.facs_pool:
            factor_data = self.load_factor(bgn_date, stp_date, factor_comb)
            factor_dfs.append(factor_data)
        x_data = pd.concat(factor_dfs, axis=1, ignore_index=False)
        return x_data

    def load_tst_ret_by_instru(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_ref = gen_tst_ret_db(
            instru=instru,
            db_save_root_dir=self.tst_ret_save_root_dir,
            save_id=self.test.ret.ret_class,
            rets=[self.test.ret.ret_name],
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_ref.db_save_dir,
            db_name=db_struct_ref.db_name,
            table=db_struct_ref.table,
            mode="r"
        )
        ret_data = sqldb.read_by_range(bgn_date, stp_date, value_columns=["trade_date", self.test.ret.ret_name])
        ret_data[self.test.ret.ret_name] = ret_data[self.test.ret.ret_name].astype(np.float64).fillna(np.nan)
        return ret_data

    def load_y(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            instru_data = self.load_tst_ret_by_instru(instru=instru, bgn_date=bgn_date, stp_date=stp_date)
            instru_data["instrument"] = instru
            dfs.append(instru_data)
        ret_data = pd.concat(dfs, axis=0, ignore_index=True)
        ret_data = ret_data.set_index(self.XY_INDEX).sort_index()
        return ret_data

    def load_sector_available(self) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_avlb.db_save_dir,
            db_name=self.db_struct_avlb.db_name,
            table=self.db_struct_avlb.table,
            mode="r"
        )
        ret_data = sqldb.read(value_columns=["trade_date", "instrument", "sectorL1"])
        ret_data.rename(columns={"sectorL1": "sector"}, inplace=True)
        sec_avlb_data = ret_data.query(f"sector == '{self.test.sector}'")
        return sec_avlb_data.set_index(self.XY_INDEX)

    @staticmethod
    def truncate_data_by_date(raw_data: pd.DataFrame, bgn_date: str, stp_date: str) -> pd.DataFrame:
        new_data = raw_data.query(f"trade_date >= '{bgn_date}' & trade_date < '{stp_date}'")
        return new_data

    @staticmethod
    def filter_by_sector(data: pd.DataFrame, sector_avlb_data: pd.DataFrame) -> pd.DataFrame:
        new_data = pd.merge(
            left=sector_avlb_data, right=data,
            left_index=True, right_index=True,
            how="inner"
        ).drop(labels="sector", axis=1)
        return new_data

    @staticmethod
    def aligned_xy(x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
        aligned_data = pd.merge(left=x_data, right=y_data, left_index=True, right_index=True, how="inner")
        s0, s1, s2 = len(x_data), len(y_data), len(aligned_data)
        if s0 == s1 == s2:
            return aligned_data
        else:
            logger.error(
                f"Length of X = {SFY(s0)}, Length of y = {SFY(s1)}, Length of aligned (X,y) = {SFY(s2)}"
            )
            raise ValueError("(X,y) have different lengths")

    @staticmethod
    def drop_and_fill_nan(aligned_data: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        idx_null = aligned_data.isnull()
        nan_data = aligned_data[idx_null.any(axis=1)]
        if not nan_data.empty:
            # keep rows where nan prop is <= threshold
            filter_nan = (idx_null.sum(axis=1) / aligned_data.shape[1]) <= threshold
            return aligned_data[filter_nan].fillna(0)
        return aligned_data

    def get_X_y(self, aligned_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return aligned_data[self.x_cols], aligned_data[self.y_col]

    def core(self, x_data: pd.DataFrame, y_data: pd.Series, trade_date: str, verbose: bool) -> TFactorNames:
        raise NotImplementedError

    def get_factor_class_and_neu_tag(self, factor_names: TFactorNames) -> tuple[list[TFactorClass], list[int]]:
        fac_cls = [self.mapper_name_to_class[n] for n in factor_names]
        is_neu = [1 if n.endswith("NEU") else 0 for n in factor_names]
        return fac_cls, is_neu

    def get_selected_feats(
            self, trade_date: str, factor_class: list[TFactorClass], factor_names: TFactorNames, is_neu: list[int],
    ) -> pd.DataFrame:
        if factor_names:
            selected_feats = pd.DataFrame(
                {
                    "trade_date": trade_date,
                    "factor_class": factor_class,
                    "factor_names": factor_names,
                    "is_neu": is_neu,
                }
            )
            return selected_feats
        else:
            logger.error(
                f"No features are selected @ {SFG(trade_date)} for {self.test.sector} {self.test.trn_win} {self.test.ret.ret_name}"
            )
            raise ValueError("No features are selected")

    def load_all_data(
            self,
            head_model_update_day: str,
            tail_model_update_day: str,
            calendar: CCalendar,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trn_b_date = calendar.get_next_date(head_model_update_day, shift=-self.test.ret.shift - self.test.trn_win + 1)
        trn_e_date = calendar.get_next_date(tail_model_update_day, shift=-self.test.ret.shift)
        trn_s_date = calendar.get_next_date(trn_e_date, shift=1)
        all_x_data, all_y_data = self.load_x(trn_b_date, trn_s_date), self.load_y(trn_b_date, trn_s_date)
        return all_x_data, all_y_data

    def select(
            self, model_update_day: str, aligned_data: pd.DataFrame, calendar: CCalendar, verbose: bool,
    ) -> pd.DataFrame:
        model_update_month = model_update_day[0:6]
        trn_b_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift - self.test.trn_win + 1)
        trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
        trn_aligned_data = aligned_data.query(f"trade_date >= '{trn_b_date}' & trade_date <= '{trn_e_date}'")
        x, y = self.get_X_y(aligned_data=trn_aligned_data)
        factor_names = self.core(x_data=x, y_data=y, trade_date=trn_e_date, verbose=verbose)
        factor_class, is_neu = self.get_factor_class_and_neu_tag(factor_names=factor_names)
        selected_feats = self.get_selected_feats(trn_e_date, factor_class, factor_names, is_neu)
        if verbose:
            logger.info(
                f"Feature selection @ {SFG(model_update_day)}, "
                f"factor selected @ {SFG(trn_e_date)}, "
                f"using train data @ [{SFG(trn_b_date)},{SFG(trn_e_date)}], "
                f"save as {SFG(model_update_month)}"
            )
        return selected_feats

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, stp_date=stp_date)
        sec_avlb_data = self.load_sector_available()
        all_x_data, all_y_data = self.load_all_data(
            head_model_update_day=model_update_days[0],
            tail_model_update_day=model_update_days[-1],
            calendar=calendar
        )
        sec_x_data = self.filter_by_sector(all_x_data, sec_avlb_data)
        sec_y_data = self.filter_by_sector(all_y_data, sec_avlb_data)
        aligned_data = self.aligned_xy(sec_x_data, sec_y_data)
        aligned_data = self.drop_and_fill_nan(aligned_data)
        selected_features: list[pd.DataFrame] = []
        for model_update_day in model_update_days:
            slc_feats = self.select(model_update_day, aligned_data, calendar, verbose)
            if not slc_feats.empty:
                selected_features.append(slc_feats)
        new_data = pd.concat(selected_features, axis=0, ignore_index=True)
        self.save(new_data=new_data)
        return 0


class CFeatSlcMutInf(CFeatSlc):
    def __init__(
            self,
            threshold: float,
            min_feats: int,
            test: CTestFtSlc,
            feat_slc_save_root_dir: str,
            tst_ret_save_root_dir: str,
            db_struct_avlb: CDbStruct,
            universe: TUniverse,
            facs_pool: TFactorsPool,
    ):
        self.threshold = threshold
        self.min_feats = min_feats
        super().__init__(
            test=test,
            feat_slc_save_root_dir=feat_slc_save_root_dir,
            tst_ret_save_root_dir=tst_ret_save_root_dir,
            db_struct_avlb=db_struct_avlb,
            universe=universe,
            facs_pool=facs_pool,
        )

    def core(self, x_data: pd.DataFrame, y_data: pd.Series, trade_date: str, verbose: bool) -> TFactorNames:
        __minimum_score = 1e-4
        importance = mutual_info_regression(X=x_data, y=y_data, random_state=self.RANDOM_STATE)
        feat_importance = pd.Series(data=importance, index=x_data.columns).sort_values(ascending=False)
        # if False:
        #     corr = [np.corrcoef(x_data[col], y_data)[0, 1] for col in x_data.columns]
        #     feat_corr = pd.Series(data=corr, index=x_data.columns).sort_values(ascending=False)
        #     df = pd.DataFrame({"mutual_info": feat_importance, "corr": feat_corr}).sort_values(
        #         by=["mutual_info", "corr"], ascending=False
        #     )

        if len(available_feats := feat_importance[feat_importance >= __minimum_score]) < self.min_feats:
            return [TFactorName(z) for z in available_feats.index]

        t, i = self.threshold, 0
        while len(selected_feats := feat_importance[feat_importance >= t]) < self.min_feats:
            t, i = t * 0.8, i + 1
        if i > 0 and verbose:
            logger.info(
                f"After {SFY(i)} times iteration {SFY(f'{len(selected_feats):>2d}')} features are selected, "
                f"{SFY(self.test.sector)}-{SFY(self.test.trn_win)}-{SFY(trade_date)}-{SFY(self.test.ret.desc)}"
            )
        return [TFactorName(z) for z in selected_feats.index]


def process_for_feature_selection(
        threshold: float,
        min_feats: int,
        test: CTestFtSlc,
        feat_slc_save_root_dir: str,
        tst_ret_save_root_dir: str,
        db_struct_avlb: CDbStruct,
        universe: TUniverse,
        facs_pool: TFactorsPool,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        verbose: bool,
):
    selector = CFeatSlcMutInf(
        threshold=threshold,
        min_feats=min_feats,
        test=test,
        feat_slc_save_root_dir=feat_slc_save_root_dir,
        tst_ret_save_root_dir=tst_ret_save_root_dir,
        db_struct_avlb=db_struct_avlb,
        universe=universe,
        facs_pool=facs_pool,
    )
    selector.main(bgn_date=bgn_date, stp_date=stp_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def main_feature_selection(
        threshold: float,
        min_feats: int,
        tests: list[CTestFtSlc],
        feat_slc_save_root_dir: str,
        tst_ret_save_root_dir: str,
        db_struct_avlb: CDbStruct,
        universe: TUniverse,
        facs_pool: TFactorsPool,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
        verbose: bool,
):
    desc = "[INF] Selecting features ..."
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for test in tests:
                    pool.apply_async(
                        process_for_feature_selection,
                        kwds={
                            "threshold": threshold,
                            "min_feats": min_feats,
                            "test": test,
                            "feat_slc_save_root_dir": feat_slc_save_root_dir,
                            "tst_ret_save_root_dir": tst_ret_save_root_dir,
                            "db_struct_avlb": db_struct_avlb,
                            "universe": universe,
                            "facs_pool": facs_pool,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                            "verbose": verbose,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for test in track(tests, description=desc):
            # for test in tests:
            process_for_feature_selection(
                threshold=threshold,
                min_feats=min_feats,
                test=test,
                feat_slc_save_root_dir=feat_slc_save_root_dir,
                tst_ret_save_root_dir=tst_ret_save_root_dir,
                db_struct_avlb=db_struct_avlb,
                universe=universe,
                facs_pool=facs_pool,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0


def get_feature_selection_tests(trn_wins: list[int], sectors: list[str], rets: list[CRet]) -> list[CTestFtSlc]:
    tests: list[CTestFtSlc] = []
    for trn_win, sector, ret in product(trn_wins, sectors, rets):
        test = CTestFtSlc(
            trn_win=trn_win,
            sector=sector,
            ret=ret,
        )
        tests.append(test)
    return tests
