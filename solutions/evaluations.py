import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from rich.progress import track, Progress
from husfort.qutility import error_handler, check_and_makedirs, SFG
from husfort.qevaluation import CNAV
from husfort.qsqlite import CMgrSqlDb, CDbStruct
from husfort.qplot import CPlotLines
from solutions.shared import gen_nav_db, group_sim_args, group_sim_args_by_sector
from typedef import CSimArgs
from typedef import TSimArgsPriKey, TSimArgsSecKey
from typedef import TSimArgsPriKeyBySec, TSimArgsSecKeyBySec, TPid

"""
Part I: Basic class
"""


class CEvl:
    def __init__(self, db_struct_nav: CDbStruct):
        self.db_struct_nav = db_struct_nav

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_nav.db_save_dir,
            db_name=self.db_struct_nav.db_name,
            table=self.db_struct_nav.table,
            mode="r"
        )
        nav_data = sqldb.read_by_range(bgn_date, stp_date)
        return nav_data

    def add_arguments(self, res: dict):
        raise NotImplementedError

    def get_ret(self, bgn_date: str, stp_date: str) -> pd.Series:
        """

        :param bgn_date:
        :param stp_date:
        :return: a pd.Series, with string index
        """
        nav_data = self.load(bgn_date, stp_date)
        ret_srs = nav_data.set_index("trade_date")["net_ret"]
        return ret_srs

    def main(self, bgn_date: str, stp_date: str) -> dict:
        indicators = ("hpr", "retMean", "retStd", "retAnnual", "volAnnual", "sharpe", "calmar", "mdd")
        ret_srs = self.get_ret(bgn_date, stp_date)
        nav = CNAV(ret_srs, input_type="RET")
        nav.cal_all_indicators()
        res = nav.to_dict()
        res = {k: v for k, v in res.items() if k in indicators}
        self.add_arguments(res)
        return res


class CEvlMdl(CEvl):
    def __init__(self, sim_args: CSimArgs, simu_mdl_dir: str):
        self.sim_args = sim_args
        db_struct_nav = gen_nav_db(db_save_dir=simu_mdl_dir, save_id=sim_args.sim_id)
        super().__init__(db_struct_nav)

    def add_arguments(self, res: dict):
        ret_class, trn_win, model_desc, sector, unique_id, ret_name, tgt_ret_name = self.sim_args.sim_id.split(".")
        other_arguments = {
            "ret_class": ret_class,
            "trn_win": trn_win,
            "model_desc": model_desc,
            "sector": sector,
            "unique_id": unique_id,
            "ret_name": ret_name,
            "ret_tgt": tgt_ret_name[1:],
        }
        res.update(other_arguments)
        return 0


def process_for_evl_mdl(sim_args: CSimArgs, simu_mdl_dir: str, bgn_date: str, stp_date: str) -> dict:
    s = CEvlMdl(sim_args, simu_mdl_dir)
    return s.main(bgn_date, stp_date)


def main_eval_mdl(
        sim_args_list: list[CSimArgs],
        simu_mdl_dir: str,
        bgn_date: str,
        stp_date: str,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Calculating evaluations"
    evl_sims: list[dict] = []
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_args_list))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                jobs = []
                for sim_args in sim_args_list:
                    job = pool.apply_async(
                        process_for_evl_mdl,
                        args=(sim_args, simu_mdl_dir, bgn_date, stp_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            evl_sims = [job.get() for job in jobs]
    else:
        for sim_args in track(sim_args_list, description=desc):
            evl = process_for_evl_mdl(sim_args, simu_mdl_dir, bgn_date, stp_date)
            evl_sims.append(evl)

    evl_data = pd.DataFrame(evl_sims)
    evl_data["sharpe_plus_calmar"] = evl_data["sharpe"] + evl_data["calmar"]
    evl_data = evl_data.sort_values(by="sharpe_plus_calmar", ascending=False)
    evl_data.insert(loc=0, column="calmar", value=evl_data.pop("calmar"))
    evl_data.insert(loc=0, column="sharpe", value=evl_data.pop("sharpe"))
    evl_data.insert(loc=0, column="sharpe_plus_calmar", value=evl_data.pop("sharpe_plus_calmar"))
    evl_data.insert(loc=0, column="unique_id", value=evl_data.pop("unique_id"))

    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda z: f"{z:.3f}")

    # --- print best models for each sector
    for (ret_name, sector), ret_name_data in evl_data.groupby(by=["ret_name", "sector"]):
        print("\n" + "-" * 180) if sector == "AGR" else print("-" * 180)
        print(f"{SFG(ret_name)}-{SFG(sector)} models with Best sharpe")  # type:ignore
        print(ret_name_data.head(3))

    # --- print sector models picked for portfolio
    print("\n" + "-" * 180)
    print(f"{SFG('sector models picked for portfolio')}")
    for (ret_name, sector), ret_name_data in evl_data.groupby(by=["ret_name", "sector"]):
        print("\n") if sector == "AGR" else 0
        uid = ret_name_data["unique_id"].iloc[0]
        sharpe, calmar = ret_name_data["sharpe"].iloc[0], ret_name_data["calmar"].iloc[0]
        sharpe_plus_calmar = ret_name_data["sharpe_plus_calmar"].iloc[0]
        win, mdl_desc = ret_name_data["trn_win"].iloc[0], ret_name_data["model_desc"].iloc[0]
        ret_tgt = ret_name_data["ret_tgt"].iloc[0]
        print(
            f"\"{uid}\", # {sector} "
            f"Sharpe = {sharpe:.3f}, Calmar = {calmar:.3f}, Sharpe + Calmar = {sharpe_plus_calmar:.3f}, "
            f"{win}, {mdl_desc}, {ret_tgt}"
        )
    return 0


"""
Part II: Plot-Single
"""


def process_for_plot(
        group_key: TSimArgsPriKey,
        sub_grouped_sim_args: dict[TSimArgsSecKey, CSimArgs],
        simu_mdl_dir: str,
        fig_save_dir: str,
        bgn_date: str,
        stp_date: str,
):
    ret_data: dict[str, pd.Series] = {}
    for (sector, unique_id), sim_args in sub_grouped_sim_args.items():
        s = CEvlMdl(sim_args, simu_mdl_dir)
        ret_data[f"{sector}-{unique_id}"] = s.get_ret(bgn_date, stp_date)
    ret_df = pd.DataFrame(ret_data).fillna(0)
    nav_df = (ret_df + 1).cumprod()
    fig_name = "-".join(group_key)

    artist = CPlotLines(
        plot_data=nav_df,
        fig_name=fig_name,
        fig_save_dir=fig_save_dir,
        fig_save_type="jpg",
        colormap="jet",
    )
    artist.plot()
    artist.save()
    return 0


def main_plot_sims(
        sim_args_list: list[CSimArgs],
        simu_mdl_dir: str,
        eval_mdl_dir: str,
        bgn_date: str,
        stp_date: str,
        call_multiprocess: bool,
        processes: int,
):
    check_and_makedirs(fig_save_dir := os.path.join(eval_mdl_dir, "plot"))
    desc = "Plotting nav ..."
    grouped_sim_args = group_sim_args(sim_args_list=sim_args_list)
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(grouped_sim_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for group_key, sub_grouped_sim_args in grouped_sim_args.items():
                    pool.apply_async(
                        process_for_plot,
                        args=(group_key, sub_grouped_sim_args, simu_mdl_dir, fig_save_dir, bgn_date, stp_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for group_key, sub_grouped_sim_args in track(grouped_sim_args.items(), description=desc):
            process_for_plot(group_key, sub_grouped_sim_args, simu_mdl_dir, fig_save_dir, bgn_date, stp_date)
    return 0


def process_for_plot_by_sector(
        sector: TSimArgsPriKeyBySec,
        sector_sim_args: dict[TSimArgsSecKeyBySec, CSimArgs],
        simu_mdl_dir: str,
        fig_save_dir: str,
        bgn_date: str,
        stp_date: str,
        top: int = 10,
):
    ret_data: dict[str, pd.Series] = {}
    for sub_key, sim_arg in sector_sim_args.items():
        sub_id = ".".join(sub_key)
        s = CEvlMdl(sim_arg, simu_mdl_dir)
        ret_data[sub_id] = s.get_ret(bgn_date, stp_date)
    ret_df = pd.DataFrame(ret_data).fillna(0)

    # selected top sharpe ratio for sector
    mu = ret_df.mean()
    sd = ret_df.std()
    sharpe = mu / sd * np.sqrt(250)
    selected_sub_ids = sharpe.sort_values(ascending=False).index[0:top]
    selected_ret_df = ret_df[selected_sub_ids]

    nav_df = (selected_ret_df + 1).cumprod()
    artist = CPlotLines(
        plot_data=nav_df,
        fig_name=sector,
        fig_save_dir=fig_save_dir,
        fig_save_type="jpg",
        colormap="jet",
    )
    artist.plot()
    artist.save()
    return 0


def main_plot_sims_by_sector(
        sim_args_list: list[CSimArgs],
        simu_mdl_dir: str,
        eval_mdl_dir: str,
        bgn_date: str,
        stp_date: str,
        call_multiprocess: bool,
        processes: int,
):
    check_and_makedirs(fig_save_dir := os.path.join(eval_mdl_dir, "plot-by-sector"))
    grouped_sim_args = group_sim_args_by_sector(sim_args_list=sim_args_list)
    desc = "Plotting nav by sector ..."
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(grouped_sim_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for sector, sector_sim_args in grouped_sim_args.items():
                    pool.apply_async(
                        process_for_plot_by_sector,
                        args=(sector, sector_sim_args, simu_mdl_dir, fig_save_dir, bgn_date, stp_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for sector, sector_sim_args in track(grouped_sim_args.items(), description=desc):
            process_for_plot_by_sector(sector, sector_sim_args, simu_mdl_dir, fig_save_dir, bgn_date, stp_date)
    return 0


"""
Part III: Evaluate and plot portfolios
"""


class CEvlPfo(CEvl):
    def __init__(self, portfolio_id: str, simu_pfo_dir: str):
        self.portfolio_id = portfolio_id
        db_struct_nav = gen_nav_db(db_save_dir=simu_pfo_dir, save_id=portfolio_id)
        super().__init__(db_struct_nav=db_struct_nav)

    def add_arguments(self, res: dict):
        other_arguments = {"portfolioId": self.portfolio_id}
        res.update(other_arguments)
        return 0


def process_for_evl_portfolio(portfolio_id: str, simu_pfo_dir: str, bgn_date: str, stp_date: str) -> dict:
    s = CEvlPfo(portfolio_id, simu_pfo_dir)
    return s.main(bgn_date, stp_date)


def main_eval_portfolios(
        portfolios: list[TPid],
        simu_pfo_dir: str,
        bgn_date: str,
        stp_date: str,
        call_multiprocess: bool,
        processes: int,
):
    evl_sims: list[dict] = []
    desc = "Calculating evaluations"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(portfolios))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                jobs = []
                for portfolio_id in portfolios:
                    job = pool.apply_async(
                        process_for_evl_portfolio,
                        args=(portfolio_id, simu_pfo_dir, bgn_date, stp_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            evl_sims = [job.get() for job in jobs]
    else:
        for portfolio_id in track(portfolios, description=desc):
            evl = process_for_evl_portfolio(portfolio_id, simu_pfo_dir, bgn_date, stp_date)
            evl_sims.append(evl)

    evl_data = pd.DataFrame(evl_sims)
    evl_data["sharpe_plus_calmar"] = evl_data["sharpe"] + evl_data["calmar"]
    evl_data = evl_data.sort_values(by="sharpe_plus_calmar", ascending=False)
    evl_data.insert(loc=0, column="calmar", value=evl_data.pop("calmar"))
    evl_data.insert(loc=0, column="sharpe", value=evl_data.pop("sharpe"))
    evl_data.insert(loc=0, column="sharpe_plus_calmar", value=evl_data.pop("sharpe_plus_calmar"))
    evl_data.insert(loc=0, column="portfolioId", value=evl_data.pop("portfolioId"))
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda z: f"{z:.3f}")
    print("Portfolios performance")
    print(evl_data)

    desc_data = evl_data.set_index("portfolioId")
    p00_sharpe, p00_calmar, p00_sum = desc_data.at["P00", "sharpe"], desc_data.at["P00", "calmar"], desc_data.at[
        "P00", "sharpe_plus_calmar"]
    p01_sharpe, p01_calmar, p01_sum = desc_data.at["P01", "sharpe"], desc_data.at["P01", "calmar"], desc_data.at[
        "P01", "sharpe_plus_calmar"]
    p0g_sharpe, p0g_calmar, p0g_sum = desc_data.at["P-OMEGA", "sharpe"], desc_data.at["P-OMEGA", "calmar"], \
        desc_data.at["P-OMEGA", "sharpe_plus_calmar"]
    summary = f"P00=({p00_sharpe:.3f},{p00_calmar:.3f},{p00_sum:.3f}), " \
              f"P01=({p01_sharpe:.3f},{p01_calmar:.3f},{p01_sum:.3f}), " \
              f"P-OMEGA=({p0g_sharpe:.3f},{p0g_calmar:.3f},{p0g_sum:.3f})"
    print(summary)
    return 0


def main_plot_portfolios(
        portfolios: list[TPid],
        simu_pfo_dir: str,
        eval_pfo_dir: str,
        bgn_date: str,
        stp_date: str,
):
    check_and_makedirs(fig_save_dir := os.path.join(eval_pfo_dir, "plot-by-portfolio"))
    ret = {}
    for portfolio_id in track(portfolios, description="Plot simulations for portfolios"):
        s = CEvlPfo(portfolio_id, simu_pfo_dir)
        ret[portfolio_id] = s.get_ret(bgn_date, stp_date)
    ret_df = pd.DataFrame(ret).fillna(0)
    nav_df = (ret_df + 1).cumprod()

    artist = CPlotLines(
        plot_data=nav_df,
        fig_name="portfolios_nav",
        fig_save_dir=fig_save_dir,
        fig_save_type="pdf",
        line_color=["#0000CD", "#DC143C", "#DAA520"],
        line_style=["-.", "-.", "-"],
        line_width=1.5,
    )
    artist.plot()
    artist.save()
    return 0
