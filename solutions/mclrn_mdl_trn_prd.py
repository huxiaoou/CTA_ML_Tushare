import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import skops.io as sio
import lightgbm as lgb
import xgboost as xgb
from loguru import logger
from rich.progress import track, Progress
from sklearn.linear_model import Ridge
from husfort.qutility import qtimer, SFG, SFY, check_and_makedirs, error_handler
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from typedef import TUniverse, TReturnName
from typedef import TFactorNames, TFactorComb, TFactorsPool
from typedef import CTestMdl, CTestFtSlc
from solutions.shared import gen_fac_db, gen_tst_ret_db, gen_prdct_db
from solutions.feature_selection import CFeatSlcReaderAndWriter

"""
Part II: Class for Machine Learning
"""


class CMclrn:
    XY_INDEX = ["trade_date", "instrument"]
    RANDOM_STATE = 0

    def __init__(
            self,
            using_instru: bool,
            test: CTestMdl,
            tst_ret_save_root_dir: str,
            factors_by_instru_dir: str,
            neutral_by_instru_dir: str,
            db_struct_avlb: CDbStruct,
            mclrn_mdl_dir: str,
            mclrn_prd_dir: str,
            universe: TUniverse,
    ):
        self.using_instru = using_instru
        self.prototype = NotImplemented
        self.fitted_estimator = NotImplemented

        self.test = test
        self.facs_pool: TFactorsPool = []
        self.tst_ret_save_root_dir = tst_ret_save_root_dir
        self.factors_by_instru_dir = factors_by_instru_dir
        self.neutral_by_instru_dir = neutral_by_instru_dir
        self.db_struct_avlb = db_struct_avlb
        self.mclrn_mdl_dir = mclrn_mdl_dir
        self.mclrn_prd_dir = mclrn_prd_dir
        self.universe = universe

    @property
    def x_cols(self) -> TFactorNames:
        ns = []
        for _, n, _ in self.facs_pool:
            ns.extend(n)
        return ns

    @property
    def y_col(self) -> TReturnName:
        return self.test.ret.ret_name

    def reset_estimator(self):
        self.fitted_estimator = None
        return 0

    def get_slc_facs_pool(self, trade_date: str) -> TFactorsPool:
        raise NotImplementedError

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

    def get_X(self, x_data: pd.DataFrame) -> pd.DataFrame:
        return x_data[self.x_cols]

    def fit_estimator(self, x_data: pd.DataFrame, y_data: pd.Series):
        if self.using_instru:
            x, y = x_data.reset_index(level="instrument"), y_data
            x["instrument"] = x["instrument"].astype("category")
        else:
            x, y = x_data.values, y_data.values
        self.fitted_estimator = self.prototype.fit(x, y)
        return 0

    def save_model(self, month_id: str):
        model_file = f"{self.test.save_tag_mdl}.skops"
        check_and_makedirs(month_dir := os.path.join(self.mclrn_mdl_dir, month_id))
        model_path = os.path.join(month_dir, model_file)
        sio.dump(self.fitted_estimator, model_path)
        return 0

    def load_model(self, month_id: str, verbose: bool) -> bool:
        model_file = f"{self.test.save_tag_mdl}.skops"
        model_path = os.path.join(self.mclrn_mdl_dir, month_id, model_file)
        if os.path.exists(model_path):
            self.fitted_estimator = sio.load(
                model_path,
                trusted=['collections.defaultdict', 'lightgbm.basic.Booster', 'lightgbm.sklearn.LGBMRegressor'],
            )
            return True
        else:
            if verbose:
                logger.info(f"No model file for {SFY(self.test.save_tag_mdl)} at {SFY(int(month_id))}")
            return False

    def apply_estimator(self, x_data: pd.DataFrame) -> pd.Series:
        if self.using_instru:
            x = x_data.reset_index(level="instrument")
            x["instrument"] = x["instrument"].astype("category")
        else:
            x = x_data.values
        pred = self.fitted_estimator.predict(X=x)  # type:ignore
        return pd.Series(data=pred, name=self.y_col, index=x_data.index)

    def train(self, model_update_day: str, sec_avlb_data: pd.DataFrame, calendar: CCalendar, verbose: bool):
        model_update_month = model_update_day[0:6]
        trn_b_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift - self.test.trn_win + 1)
        trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
        trn_s_date = calendar.get_next_date(trn_e_date, shift=1)
        self.facs_pool = self.get_slc_facs_pool(trade_date=trn_e_date)
        sec_avlb_data_m = self.truncate_data_by_date(sec_avlb_data, trn_b_date, trn_s_date)
        x_data, y_data = self.load_x(trn_b_date, trn_s_date), self.load_y(trn_b_date, trn_s_date)
        x_data, y_data = self.filter_by_sector(x_data, sec_avlb_data_m), self.filter_by_sector(y_data, sec_avlb_data_m)
        aligned_data = self.aligned_xy(x_data, y_data)
        aligned_data = self.drop_and_fill_nan(aligned_data)
        x, y = self.get_X_y(aligned_data=aligned_data)
        self.fit_estimator(x_data=x, y_data=y)
        self.save_model(month_id=model_update_month)
        if verbose:
            logger.info(
                f"Train model @ {SFG(model_update_day)}, "
                f"factor selected @ {SFG(trn_e_date)}, "
                f"using train data @ [{SFG(trn_b_date)},{SFG(trn_e_date)}], "
                f"save as {SFG(model_update_month)}"
            )
        return 0

    def process_trn(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        sec_avlb_data = self.load_sector_available()
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, stp_date=stp_date)
        for model_update_day in model_update_days:
            self.train(model_update_day, sec_avlb_data, calendar, verbose)
        return 0

    def predict(
            self,
            prd_month_id: str,
            prd_month_days: list[str],
            sec_avlb_data: pd.DataFrame,
            calendar: CCalendar,
            verbose: bool,
    ) -> pd.Series:
        trn_month_id = calendar.get_next_month(prd_month_id, -1)
        self.reset_estimator()
        if self.load_model(month_id=trn_month_id, verbose=verbose):
            model_update_day = calendar.get_last_day_of_month(trn_month_id)
            trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
            prd_b_date, prd_e_date = prd_month_days[0], prd_month_days[-1]
            prd_s_date = calendar.get_next_date(prd_e_date, shift=1)
            self.facs_pool = self.get_slc_facs_pool(trade_date=trn_e_date)
            sec_avlb_data_m = self.truncate_data_by_date(sec_avlb_data, prd_b_date, prd_s_date)
            x_data = self.load_x(prd_b_date, prd_s_date)
            x_data = self.filter_by_sector(x_data, sec_avlb_data_m)
            x_data = self.drop_and_fill_nan(x_data)
            x_data = self.get_X(x_data=x_data)
            y_h_data = self.apply_estimator(x_data=x_data)
            if verbose:
                logger.info(
                    f"Call model @ {SFG(model_update_day)}, "
                    f"factor selected @ {SFG(trn_e_date)}, "
                    f"prediction @ [{SFG(prd_b_date)},{SFG(prd_e_date)}], "
                    f"load model from {SFG(trn_month_id)}"
                )
            return y_h_data.astype(np.float64)
        else:
            return pd.Series(dtype=np.float64)

    def process_prd(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool) -> pd.DataFrame:
        sec_avlb_data = self.load_sector_available()
        months_groups = calendar.split_by_month(dates=calendar.get_iter_list(bgn_date, stp_date))
        pred_res: list[pd.Series] = []
        for prd_month_id, prd_month_days in months_groups.items():
            month_prediction = self.predict(prd_month_id, prd_month_days, sec_avlb_data, calendar, verbose)
            pred_res.append(month_prediction)
        prediction = pd.concat(pred_res, axis=0, ignore_index=False)
        prediction.index = pd.MultiIndex.from_tuples(prediction.index, names=self.XY_INDEX)
        sorted_prediction = prediction.reset_index().sort_values(["trade_date", "instrument"])
        return sorted_prediction

    def process_save_prediction(self, prediction: pd.DataFrame, calendar: CCalendar):
        db_struct_prdct = gen_prdct_db(self.mclrn_prd_dir, self.test)
        check_and_makedirs(db_struct_prdct.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_prdct.db_save_dir,
            db_name=db_struct_prdct.db_name,
            table=db_struct_prdct.table,
            mode="a",
        )
        if sqldb.check_continuity(incoming_date=prediction["trade_date"].iloc[0], calendar=calendar) == 0:
            sqldb.update(update_data=prediction)
        return 0

    def main_mclrn_model(self, bgn_date: str, stp_date: str, calendar: CCalendar, verbose: bool):
        self.process_trn(bgn_date, stp_date, calendar, verbose)
        prediction = self.process_prd(bgn_date, stp_date, calendar, verbose)
        self.process_save_prediction(prediction, calendar)
        return 0


class CMclrnFromFeatureSelection(CMclrn):
    def __init__(
            self,
            using_instru: bool,
            test: CTestMdl,
            tst_ret_save_root_dir: str,
            factors_by_instru_dir: str,
            neutral_by_instru_dir: str,
            feat_slc_save_root_dir: str,
            db_struct_avlb: CDbStruct,
            mclrn_mdl_dir: str,
            mclrn_prd_dir: str,
            universe: TUniverse,
    ):
        super().__init__(
            using_instru=using_instru,
            test=test,
            tst_ret_save_root_dir=tst_ret_save_root_dir,
            factors_by_instru_dir=factors_by_instru_dir,
            neutral_by_instru_dir=neutral_by_instru_dir,
            db_struct_avlb=db_struct_avlb,
            mclrn_mdl_dir=mclrn_mdl_dir,
            mclrn_prd_dir=mclrn_prd_dir,
            universe=universe,
        )
        test_slc_fac = CTestFtSlc(trn_win=test.trn_win, sector=test.sector, ret=test.ret)
        self.slc_fac_reader = CFeatSlcReaderAndWriter(
            test=test_slc_fac,
            feat_slc_save_root_dir=feat_slc_save_root_dir,
            tst_ret_save_root_dir=tst_ret_save_root_dir,
        )

    def get_slc_facs_pool(self, trade_date: str) -> TFactorsPool:
        return self.slc_fac_reader.get_slc_facs_pool(
            trade_date=trade_date,
            factors_by_instru_dir=self.factors_by_instru_dir,
            neutral_by_instru_dir=self.neutral_by_instru_dir,
        )


class CMclrnRidge(CMclrnFromFeatureSelection):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(using_instru=False, **kwargs)
        self.prototype = Ridge(alpha=alpha, fit_intercept=False)


class CMclrnLGBM(CMclrnFromFeatureSelection):
    def __init__(
            self,
            boosting_type: str,
            metric: str,
            max_depth: int,
            num_leaves: int,
            learning_rate: float,
            n_estimators: int,
            min_child_samples: int,
            max_bin: int,
            **kwargs,
    ):
        super().__init__(using_instru=True, **kwargs)
        self.prototype = lgb.LGBMRegressor(
            boosting_type=boosting_type,
            metric=metric,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            max_bin=max_bin,
            # other fixed parameters
            force_row_wise=True,
            verbose=-1,
            random_state=self.RANDOM_STATE,
            device_type="gpu",
        )


class CMclrnXGB(CMclrnFromFeatureSelection):
    def __init__(
            self,
            booster: str,
            n_estimators: int,
            max_depth: int,
            max_leaves: int,
            grow_policy: str,
            learning_rate: float,
            objective: str,
            **kwargs,
    ):
        super().__init__(using_instru=False, **kwargs)
        self.prototype = xgb.XGBRegressor(
            booster=booster,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            objective=objective,
            # other fixed parameters
            verbosity=0,
            random_state=self.RANDOM_STATE,
        )


"""
Part III: Wrapper for CMclrn

"""


def process_for_cMclrn(
        test: CTestMdl,
        tst_ret_save_root_dir: str,
        factors_by_instru_dir: str,
        neutral_by_instru_dir: str,
        feat_slc_save_root_dir: str,
        db_struct_avlb: CDbStruct,
        mclrn_mdl_dir: str,
        mclrn_prd_dir: str,
        universe: TUniverse,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        verbose: bool,
):
    x: dict[str, type[CMclrnRidge] | type[CMclrnLGBM] | type[CMclrnXGB]] = {
        "Ridge": CMclrnRidge,
        "LGBM": CMclrnLGBM,
        "XGB": CMclrnXGB,
    }
    if not (mclrn_type := x.get(test.model.model_type)):
        raise ValueError(f"model type = {test.model.model_type} is wrong")

    mclrn = mclrn_type(
        test=test,
        tst_ret_save_root_dir=tst_ret_save_root_dir,
        factors_by_instru_dir=factors_by_instru_dir,
        neutral_by_instru_dir=neutral_by_instru_dir,
        feat_slc_save_root_dir=feat_slc_save_root_dir,
        db_struct_avlb=db_struct_avlb,
        mclrn_mdl_dir=mclrn_mdl_dir,
        mclrn_prd_dir=mclrn_prd_dir,
        universe=universe,
        **test.model.model_args,
    )
    os.environ["OMP_NUM_THREADS"] = "8"
    mclrn.main_mclrn_model(bgn_date=bgn_date, stp_date=stp_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def main_train_and_predict(
        tests: list[CTestMdl],
        tst_ret_save_root_dir: str,
        factors_by_instru_dir: str,
        neutral_by_instru_dir: str,
        feat_slc_save_root_dir: str,
        db_struct_avlb: CDbStruct,
        mclrn_mdl_dir: str,
        mclrn_prd_dir: str,
        universe: TUniverse,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
        verbose: bool,
):
    desc = "Training and predicting for machine learning"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for test in tests:
                    pool.apply_async(
                        process_for_cMclrn,
                        kwds={
                            "test": test,
                            "tst_ret_save_root_dir": tst_ret_save_root_dir,
                            "factors_by_instru_dir": factors_by_instru_dir,
                            "neutral_by_instru_dir": neutral_by_instru_dir,
                            "feat_slc_save_root_dir": feat_slc_save_root_dir,
                            "db_struct_avlb": db_struct_avlb,
                            "mclrn_mdl_dir": mclrn_mdl_dir,
                            "mclrn_prd_dir": mclrn_prd_dir,
                            "universe": universe,
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
            process_for_cMclrn(
                test=test,
                tst_ret_save_root_dir=tst_ret_save_root_dir,
                factors_by_instru_dir=factors_by_instru_dir,
                neutral_by_instru_dir=neutral_by_instru_dir,
                feat_slc_save_root_dir=feat_slc_save_root_dir,
                db_struct_avlb=db_struct_avlb,
                mclrn_mdl_dir=mclrn_mdl_dir,
                mclrn_prd_dir=mclrn_prd_dir,
                universe=universe,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0
