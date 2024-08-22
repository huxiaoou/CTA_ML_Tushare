import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate data, such as macro and forex")
    arg_parser.add_argument(
        "--switch", type=str,
        choices=("available", "market", "test_return"),
        required=True
    )
    arg_parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true",
                            help="not using multiprocess, for debug. Works only when switch in ('preprocess',)")
    arg_parser.add_argument("--processes", type=int, default=None,
                            help="number of processes to be called, effective only when nomp = False")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from project_cfg import proj_cfg, db_struct_cfg
    from husfort.qlog import define_logger
    from husfort.qcalendar import CCalendar

    define_logger()

    calendar = CCalendar(proj_cfg.calendar_path)
    args = parse_args()
    bgn_date, stp_date = args.bgn, args.stp or calendar.get_next_date(args.bgn, shift=1)

    if args.switch == "available":
        from solutions.available import main_available

        main_available(
            bgn_date=bgn_date, stp_date=stp_date,
            universe=proj_cfg.universe,
            cfg_avlb_unvrs=proj_cfg.avlb_unvrs,
            db_struct_preprocess=db_struct_cfg.preprocess,
            db_struct_avlb=db_struct_cfg.available,
            calendar=calendar,
        )
    elif args.switch == "market":
        from solutions.market import main_market

        main_market(
            bgn_date=bgn_date, stp_date=stp_date,
            calendar=calendar,
            db_struct_avlb=db_struct_cfg.available,
            db_struct_mkt=db_struct_cfg.market,
            path_mkt_idx_data=proj_cfg.market_index_path,
            mkt_idxes=list(proj_cfg.mkt_idxes.values()),
            sectors=proj_cfg.const.SECTORS,
        )
    elif args.switch == "test_return":
        from solutions.test_return import CTstRetRaw, CTstRetNeu

        for lag, win in zip((proj_cfg.const.LAG, 1), (proj_cfg.const.WIN, 1)):
            # --- raw return
            tst_ret = CTstRetRaw(
                lag=lag, win=win,
                universe=list(proj_cfg.universe),
                db_tst_ret_save_dir=proj_cfg.test_return_dir,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
            tst_ret.main_test_return_raw(bgn_date, stp_date, calendar)

            # --- neutralization
            tst_ret_neu = CTstRetNeu(
                lag=lag,
                win=win,
                universe=list(proj_cfg.universe),
                db_tst_ret_save_dir=proj_cfg.test_return_dir,
                db_struct_preprocess=db_struct_cfg.preprocess,
                db_struct_avlb=db_struct_cfg.available,
            )
            tst_ret_neu.main_test_return_neu(
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
    else:
        raise ValueError(f"args.switch = {args.switch} is illegal")
