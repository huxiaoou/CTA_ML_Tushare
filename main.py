import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate data, such as macro and forex")
    arg_parser.add_argument(
        "--switch", type=str,
        choices=("available",),
        required=True
    )
    arg_parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true",
                            help="not using multiprocess, for debug. Works only when switch in ('preprocess',)")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from project_cfg import pro_cfg, db_struct_cfg
    from husfort.qlog import define_logger
    from husfort.qcalendar import CCalendar

    define_logger()

    calendar = CCalendar(pro_cfg.calendar_path)
    args = parse_args()
    bgn_date, stp_date = args.bgn, args.stp or calendar.get_next_date(args.bgn, shift=1)

    if args.switch == "available":
        raise NotImplementedError
    else:
        raise ValueError(f"args.switch = {args.switch} is illegal")
