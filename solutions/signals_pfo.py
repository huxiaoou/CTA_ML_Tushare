import multiprocessing as mp
import pandas as pd
from husfort.qsqlite import CMgrSqlDb
from rich.progress import track, Progress
from husfort.qutility import error_handler, check_and_makedirs
from husfort.qcalendar import CCalendar
from typedef import CSimArgs, CPortfolioArgs, TUniqueId
from solutions.shared import gen_sig_pfo_db


class CSignalPortfolio:
    def __init__(
            self,
            pid: str,
            target: str,
            weights: list[TUniqueId],
            portfolio_sim_args: dict[str, CSimArgs],
            signals_pfo_dir: str,
    ):
        self.pid = pid
        self.target = target
        self.weights = weights
        self.portfolio_sim_args = portfolio_sim_args
        self.db_struct_sig_pfo = gen_sig_pfo_db(db_save_root_dir=signals_pfo_dir, portfolio_id=pid)

    @staticmethod
    def load_from_sim_args(sim_args: CSimArgs, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=sim_args.db_struct_sig.db_save_dir,
            db_name=sim_args.db_struct_sig.db_name,
            table=sim_args.db_struct_sig.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    @staticmethod
    def multiply_sector_weight(df: pd.DataFrame) -> pd.DataFrame:
        _wgt = "weight"
        size = len(df)
        df[_wgt] = df[_wgt] * size
        return df

    def reformat_sig(self, sig_data: pd.DataFrame) -> pd.DataFrame:
        new_data = sig_data[["trade_date", "instrument", "weight"]].fillna(0)
        new_data_gt0 = new_data.query("abs(weight) > 0")
        new_data_adj = new_data_gt0.groupby(by="trade_date", group_keys=False).apply(self.multiply_sector_weight)
        return new_data_adj

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sec_signal_dfs: list[pd.DataFrame] = []
        for unique_id in self.weights:
            sim_args = self.portfolio_sim_args[unique_id]
            mdl_weight = self.load_from_sim_args(sim_args, bgn_date, stp_date)
            sec_signal_dfs.append(self.reformat_sig(mdl_weight))
        signal_data = pd.concat(sec_signal_dfs, axis=0, ignore_index=True)
        return signal_data

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        _wgt = "weight"
        abs_sum = df[_wgt].abs().sum()
        if abs_sum > 0:
            df[_wgt] = df[_wgt] / abs_sum
        return df

    def cal_portfolio_weights(self, signal_data: pd.DataFrame) -> pd.DataFrame:
        wgt_sum_data_norm = signal_data.groupby(by="trade_date", group_keys=False).apply(self.normalize)
        return wgt_sum_data_norm

    def save(self, new_data: pd.DataFrame, calendar: CCalendar):
        check_and_makedirs(self.db_struct_sig_pfo.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_sig_pfo.db_save_dir,
            db_name=self.db_struct_sig_pfo.db_name,
            table=self.db_struct_sig_pfo.table,
            mode="a",
        )
        if sqldb.check_continuity(incoming_date=new_data["trade_date"].iloc[0], calendar=calendar) == 0:
            sqldb.update(update_data=new_data)
        return 0

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        signal_data = self.load(bgn_date, stp_date)
        wgt_sum_data_norm = self.cal_portfolio_weights(signal_data)
        self.save(wgt_sum_data_norm, calendar)
        return 0


def process_for_cal_signal_portfolio(
        portfolio_arg: CPortfolioArgs,
        signals_pfo_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
):
    signal_portfolio = CSignalPortfolio(
        pid=portfolio_arg.pid,
        target=portfolio_arg.target,
        weights=portfolio_arg.weights,
        portfolio_sim_args=portfolio_arg.portfolio_sim_args,
        signals_pfo_dir=signals_pfo_dir,
    )
    signal_portfolio.main(bgn_date, stp_date, calendar)
    return 0


def main_signals_portfolios(
        portfolio_args: list[CPortfolioArgs],
        signals_pfo_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Combing portfolio signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(portfolio_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for portfolio_arg in portfolio_args:
                    pool.apply_async(
                        process_for_cal_signal_portfolio,
                        kwds={
                            "portfolio_arg": portfolio_arg,
                            "signals_pfo_dir": signals_pfo_dir,
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
        for portfolio_arg in track(portfolio_args, description=desc):
            process_for_cal_signal_portfolio(
                portfolio_arg=portfolio_arg,
                signals_pfo_dir=signals_pfo_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0
