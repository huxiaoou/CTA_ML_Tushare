import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qutility import qtimer, error_handler, check_and_makedirs
from husfort.qsqlite import CMgrSqlDb
from husfort.qcalendar import CCalendar
from solutions.shared import gen_prdct_db, gen_sig_mdl_db
from typedef import CTestMdl

"""
Part I: Signals from single mdl
"""


class CSignal:
    def __init__(self, input_dir: str, output_dir: str, test_mdl: CTestMdl, maw: int):
        self.test_mdl = test_mdl
        self.maw = maw  # moving average window
        self.db_struct_prd = gen_prdct_db(db_save_root_dir=input_dir, test=test_mdl)
        self.db_struct_sig = gen_sig_mdl_db(db_save_root_dir=output_dir, test=test_mdl)

    def load_input(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_bgn_date = calendar.get_next_date(bgn_date, -self.maw + 1)
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_prd.db_save_dir,
            db_name=self.db_struct_prd.db_name,
            table=self.db_struct_prd.table,
            mode="r",
        )
        prd_data = sqldb.read_by_range(bgn_date=base_bgn_date, stp_date=stp_date)
        return prd_data

    def process_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(axis=0, subset=[self.test_mdl.ret.ret_name], how="any")

    def cal_signal(self, clean_data: pd.DataFrame) -> pd.DataFrame:
        """
        params clean_data: pd.DataFrame with columns = ["trade_date", "instrument", self.test.ret.ret_name]
        return : pd.DataFrame with columns = ["trade_date", "instrument", self.test.ret.ret_name]

        """
        raise NotImplementedError

    def moving_average_signal(self, signal_data: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        pivot_data = pd.pivot_table(
            data=signal_data,
            index=["trade_date"],
            columns=["instrument"],
            values=[self.test_mdl.ret.ret_name],
        )
        instru_ma_data = pivot_data.fillna(0).rolling(window=self.maw).mean()
        truncated_data = instru_ma_data.query(f"trade_date >= '{bgn_date}'")
        normalize_data = truncated_data.div(truncated_data.abs().sum(axis=1), axis=0).fillna(0)
        stack_data = normalize_data.stack(future_stack=True).reset_index()
        return stack_data[["trade_date", "instrument", self.test_mdl.ret.ret_name]]

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        check_and_makedirs(self.db_struct_sig.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_sig.db_save_dir,
            db_name=self.db_struct_sig.db_name,
            table=self.db_struct_sig.table,
            mode="a",
        )
        if sqldb.check_continuity(bgn_date, calendar) == 0:
            input_data = self.load_input(bgn_date, stp_date, calendar)
            clean_data = self.process_nan(input_data)
            signal_data = self.cal_signal(clean_data)
            signal_data_ma = self.moving_average_signal(signal_data, bgn_date)
            sqldb.update(signal_data_ma)
        return 0


class CSignalCrsSec(CSignal):
    @staticmethod
    def map_prediction_to_signal(data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        s = [1] * int(n / 2) + [0] * (n % 2) + [-1] * int(n / 2)
        data["signal"] = s
        if (abs_sum := data["signal"].abs().sum()) > 0:
            data["signal"] = data["signal"] / abs_sum
        return data[["trade_date", "instrument", "signal"]]

    def cal_signal(self, clean_data: pd.DataFrame) -> pd.DataFrame:
        sorted_data = clean_data.sort_values(
            by=["trade_date", self.test_mdl.ret.ret_name, "instrument"], ascending=[True, False, True]
        )
        grouped_data = sorted_data.groupby(by=["trade_date"], group_keys=False)
        signal_data = grouped_data.apply(self.map_prediction_to_signal)
        signal_data.rename(mapper={"signal": self.test_mdl.ret.ret_name}, axis=1, inplace=True)
        return signal_data


def process_for_signal(
        input_dir: str,
        output_dir: str,
        test_mdl: CTestMdl,
        maw: int,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
):
    signal = CSignalCrsSec(input_dir=input_dir, output_dir=output_dir, test_mdl=test_mdl, maw=maw)
    signal.main(bgn_date, stp_date, calendar)
    return 0


@qtimer
def main_signals_models(
        test_mdls: list[CTestMdl],
        prd_save_root_dir: str,
        sig_mdl_save_root_dir: str,
        maw: int,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Translating prediction to signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(test_mdls))
            with mp.get_context("spawn").Pool(processes) as pool:
                for test_mdl in test_mdls:
                    pool.apply_async(
                        process_for_signal,
                        kwds={
                            "input_dir": prd_save_root_dir,
                            "output_dir": sig_mdl_save_root_dir,
                            "test_mdl": test_mdl,
                            "maw": maw,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for test_mdl in track(test_mdls, description=desc):
            process_for_signal(
                input_dir=prd_save_root_dir,
                output_dir=sig_mdl_save_root_dir,
                test_mdl=test_mdl,
                maw=maw,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0
