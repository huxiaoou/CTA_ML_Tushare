import multiprocessing as mp
import pandas as pd
from rich.progress import track, Progress
from husfort.qutility import qtimer, error_handler, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CMgrSqlDb
from solutions.shared import gen_nav_db
from typedef import CSimArgs


class CSim:
    def __init__(self, sim_args: CSimArgs, sim_save_dir: str):
        self.sim_args = sim_args
        self.sim_save_dir = sim_save_dir
        self.db_struct_sim = gen_nav_db(db_save_dir=sim_save_dir, save_id=sim_args.sim_id)

    def load_sig(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.sim_args.db_struct_sig.db_save_dir,
            db_name=self.sim_args.db_struct_sig.db_name,
            table=self.sim_args.db_struct_sig.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    def load_ret(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.sim_args.db_struct_ret.db_save_dir,
            db_name=self.sim_args.db_struct_ret.db_name,
            table=self.sim_args.db_struct_ret.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    def reformat_sig(self, sig_data: pd.DataFrame) -> pd.DataFrame:
        new_data = sig_data.rename(mapper={self.sim_args.sig_name: "sig"}, axis=1)
        new_data = new_data[["trade_date", "instrument", "sig"]].fillna(0)
        return new_data

    def reformat_ret(self, ret_data: pd.DataFrame) -> pd.DataFrame:
        new_data = ret_data.rename(mapper={self.sim_args.tgt_ret.ret_name: "ret"}, axis=1)
        new_data = new_data[["trade_date", "instrument", "ret"]].fillna(0)
        return new_data

    @staticmethod
    def merge_sig_and_ret(sig_data: pd.DataFrame, ret_data: pd.DataFrame) -> pd.DataFrame:
        merged_data = pd.merge(left=sig_data, right=ret_data, on=["trade_date", "instrument"], how="inner")
        return merged_data.dropna(axis=0, subset=["sig"], how="any")

    def cal_ret(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        raw_ret = merged_data.groupby(by="trade_date", group_keys=True).apply(lambda z: z["sig"] @ z["ret"])
        wgt_data = pd.pivot_table(
            data=merged_data,
            index="trade_date",
            columns="instrument",
            values="sig",
            aggfunc="mean",
        ).fillna(0)
        wgt_data_prev = wgt_data.shift(1).fillna(0)
        wgt_diff = wgt_data - wgt_data_prev
        dlt_wgt = wgt_diff.abs().sum(axis=1)
        cost = dlt_wgt * self.sim_args.cost
        net_ret = raw_ret - cost
        sim_data = pd.DataFrame({"raw_ret": raw_ret, "dlt_wgt": dlt_wgt, "cost": cost, "net_ret": net_ret})
        return sim_data

    def align_dates(self, sim_data: pd.DataFrame, bgn_date: str, calendar: CCalendar) -> pd.DataFrame:
        aligned_sim_data = sim_data.reset_index()
        aligned_sim_data["trade_date"] = aligned_sim_data["trade_date"].map(
            lambda z: calendar.get_next_date(z, self.sim_args.tgt_ret.shift)
        )
        return aligned_sim_data.query(f"trade_date >= '{bgn_date}'")

    @staticmethod
    def update_nav(aligned_sim_data: pd.DataFrame, last_nav: float) -> pd.DataFrame:
        aligned_sim_data["nav"] = (aligned_sim_data["net_ret"] + 1).cumprod() * last_nav  # type:ignore
        return aligned_sim_data

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        check_and_makedirs(self.db_struct_sim.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_sim.db_save_dir,
            db_name=self.db_struct_sim.db_name,
            table=self.db_struct_sim.table,
            mode="a",
        )
        if sqldb.check_continuity(bgn_date, calendar) == 0:
            d = 0 if sqldb.empty else 1
            iter_dates = calendar.get_iter_list(bgn_date, stp_date)
            base_bgn_date = calendar.get_next_date(iter_dates[0], shift=-self.sim_args.tgt_ret.shift - d)
            base_end_date = calendar.get_next_date(iter_dates[-1], shift=-self.sim_args.tgt_ret.shift)
            base_stp_date = calendar.get_next_date(base_end_date, shift=1)
            sig_data = self.load_sig(bgn_date=base_bgn_date, stp_date=base_stp_date)
            ret_data = self.load_ret(bgn_date=base_bgn_date, stp_date=base_stp_date)
            rft_sig_data, rft_ret_data = self.reformat_sig(sig_data), self.reformat_ret(ret_data)
            merged_data = self.merge_sig_and_ret(sig_data=rft_sig_data, ret_data=rft_ret_data)
            sim_data = self.cal_ret(merged_data=merged_data)
            aligned_sim_data = self.align_dates(sim_data, bgn_date=bgn_date, calendar=calendar)
            aligned_sim_data[["raw_ret", "net_ret"]] = aligned_sim_data[["raw_ret", "net_ret"]].fillna(0)
            new_data = self.update_nav(
                aligned_sim_data=aligned_sim_data,
                last_nav=sqldb.last_val(val="nav", val_if_none=1.0)
            )
            null_data = new_data[new_data.isnull().any(axis=1)]
            if not null_data.empty:
                raise ValueError(f"{self.sim_args.sim_id} has nan data")
            sqldb.update(update_data=new_data)
        return 0


def process_for_sim(
        sim_arg: CSimArgs,
        sim_save_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
):
    sim = CSim(sim_args=sim_arg, sim_save_dir=sim_save_dir)
    sim.main(bgn_date, stp_date, calendar)
    return 0


@qtimer
def main_simulations(
        sim_args: list[CSimArgs],
        sim_save_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Calculating simulations"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_args))
            with mp.get_context("spawn").Pool(processes) as pool:
                for sim_arg in sim_args:
                    pool.apply_async(
                        process_for_sim,
                        kwds={
                            "sim_arg": sim_arg,
                            "sim_save_dir": sim_save_dir,
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
        for sim_arg in track(sim_args, description=desc):
            process_for_sim(
                sim_arg=sim_arg,
                sim_save_dir=sim_save_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0
