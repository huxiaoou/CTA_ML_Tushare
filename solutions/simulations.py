import multiprocessing as mp
import pandas as pd
from rich.progress import track, Progress
from husfort.qutility import qtimer, error_handler, SFR, check_and_makedirs
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CMgrSqlDb
from solutions.shared import gen_tst_ret_regrp_db, gen_sig_mdl_db, gen_nav_db
from typedef import CTest, CSimArgs, CRet


def get_sim_args_from_tests(
        tests: list[CTest], cost: float, test_return_dir: str, signals_mdl_dir: str
) -> list[CSimArgs]:
    res: list[CSimArgs] = []
    for test in tests:
        if test.ret.ret_name.startswith("Opn"):
            tgt_ret_class, tgt_ret_name = "001L1", "OpnRtn001L1RAW"
        elif test.ret.ret_name.startswith("Cls"):
            tgt_ret_class, tgt_ret_name = "001L1", "ClsRtn001L1RAW"
        else:
            raise ValueError(f"ret_name = {SFR(test.ret.ret_name)} is illegal")
        tgt_ret = CRet(tgt_ret_class, tgt_ret_name, 2)

        db_struct_ret = gen_tst_ret_regrp_db(db_save_root_dir=test_return_dir, ret_name=tgt_ret_name)
        db_struct_sig = gen_sig_mdl_db(db_save_root_dir=signals_mdl_dir, test=test)
        sim_id = f"{test.save_tag_mdl}.T{tgt_ret_name}"

        sim_arg = CSimArgs(
            sim_id=sim_id,
            sig_name=test.ret.ret_name,
            tgt_ret=tgt_ret,
            db_struct_sig=db_struct_sig,
            db_struct_ret=db_struct_ret,
            cost=cost,
        )
        res.append(sim_arg)
    return res


# def group_sim_args(sim_args: list[CSimArg]) -> dict[tuple[str, str, str, str], dict[tuple[str, str], CSimArg]]:
#     grouped_sim_args: dict[tuple[str, str, str, str], dict[tuple[str, str], CSimArg]] = {}
#     for sim_arg in sim_args:
#         ret_class, trn_win, model_desc, sector, unique_id, ret_name, tgt_ret_name = sim_arg.sim_id.split(".")
#         key0, key1 = (ret_class, trn_win, model_desc, ret_name), (sector, unique_id)
#         if key0 not in grouped_sim_args:
#             grouped_sim_args[key0] = {}
#         grouped_sim_args[key0][key1] = sim_arg
#     return grouped_sim_args
#
#
# def group_sim_args_by_sector(sim_args: list[CSimArg]) -> dict[str, dict[tuple[str, str, str, str, str], CSimArg]]:
#     grouped_sim_args: dict[str, dict[tuple[str, str, str, str, str], CSimArg]] = {}
#     for sim_arg in sim_args:
#         ret_class, trn_win, model_desc, sector, unique_id, ret_name, tgt_ret_name = sim_arg.sim_id.split(".")
#         key0, key1 = sector, (ret_class, trn_win, model_desc, ret_name, unique_id)
#         if key0 not in grouped_sim_args:
#             grouped_sim_args[key0] = {}
#         grouped_sim_args[key0][key1] = sim_arg
#     return grouped_sim_args


# @dataclass(frozen=True)
# class CPortfolioArg:
#     pid: str
#     target: str
#     weights: dict[str, float]
#     portfolio_sim_args: dict[str, CSimArg]
#
#
# def get_portfolio_args(portfolios: dict[str, dict], sim_args: list[CSimArg]) -> list[CPortfolioArg]:
#     res: list[CPortfolioArg] = []
#     for portfolio_id, portfolio_cfg in portfolios.items():
#         target, weights = portfolio_cfg["target"], portfolio_cfg["weights"]
#         portfolio_sim_args = {}
#         for sim_arg in sim_args:
#             *_, unique_id, ret_name = sim_arg.sim_id.split(".")
#             if (unique_id in weights) and (ret_name == target):
#                 portfolio_sim_args[unique_id] = sim_arg
#         portfolio_arg = CPortfolioArg(portfolio_id, target, weights, portfolio_sim_args)
#         res.append(portfolio_arg)
#     return res
#
#
# def get_sim_args_from_portfolios(
#         portfolios: dict[str, dict], prefix_user: list[str], cost: float, shift: int
# ) -> list[CSimArg]:
#     res: list[CSimArg] = []
#     for portfolio_id, portfolio_cfg in portfolios.items():
#         target: str = portfolio_cfg["target"]
#         sig = CSimSig(prefix=prefix_user + ["signals", "portfolios", portfolio_id], sid=target)
#         if target.startswith("Open"):
#             ret_class, ret_name = "001L1", "OpenRtn001L1"
#             # ret_class, ret_name = "010L1", "OpenRtn010L1"
#         elif target.startswith("Close"):
#             ret_class, ret_name = "001L1", "CloseRtn001L1"
#             # ret_class, ret_name = "010L1", "CloseRtn010L1"
#         else:
#             raise ValueError(f"ret_name = {target} is illegal")
#         ret = CSimRet(prefix=prefix_user + ["Y"] + [ret_class], ret=ret_name, shift=shift)
#         sim_arg = CSimArg(sim_id=portfolio_id, sig=sig, ret=ret, cost=cost)
#         res.append(sim_arg)
#     return res


class CSim:
    def __init__(self, sim_args: CSimArgs, sim_save_dir: str):
        self.sim_args = sim_args
        self.sim_save_dir = sim_save_dir
        self.db_struct_sim = gen_nav_db(db_save_dir=sim_save_dir, sim_arg=sim_args)

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
