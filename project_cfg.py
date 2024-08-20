import yaml
from typedef import TUniverse, CInstruCfg
from typedef import CProCfg, CDbStructCfg
from husfort.qsqlite import CDbStruct, CSqlTable

# ---------- project configuration ----------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

universe: TUniverse = {k: CInstruCfg(**v) for k, v in config["universe"].items()}

pro_cfg = CProCfg(
    calendar_path=config["path"]["calendar_path"],
    root_dir=config["path"]["root_dir"],
    daily_data_root_dir=config["path"]["daily_data_root_dir"],
    db_struct_path=config["path"]["db_struct_path"],
    alternative_dir=config["path"]["alternative_dir"],
    by_instru_pos_dir=config["path"]["by_instru_pos_dir"],
    by_instru_pre_dir=config["path"]["by_instru_pre_dir"],
    universe=universe,
)

# ---------- databases structure ----------
with open(pro_cfg.db_struct_path, "r") as f:
    db_struct = yaml.safe_load(f)

db_struct_cfg = CDbStructCfg(
    macro=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["macro"]["db_name"],
        table=CSqlTable(cfg=db_struct["macro"]["table"]),
    ),
    forex=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["forex"]["db_name"],
        table=CSqlTable(cfg=db_struct["forex"]["table"]),
    ),
    fmd=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["fmd"]["db_name"],
        table=CSqlTable(cfg=db_struct["fmd"]["table"]),
    ),
    position=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["position"]["db_name"],
        table=CSqlTable(cfg=db_struct["position"]["table"]),
    ),
    basis=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["basis"]["db_name"],
        table=CSqlTable(cfg=db_struct["basis"]["table"]),
    ),
    stock=CDbStruct(
        db_save_dir=pro_cfg.root_dir,
        db_name=db_struct["stock"]["db_name"],
        table=CSqlTable(cfg=db_struct["stock"]["table"]),
    ),
    preprocess=CDbStruct(
        db_save_dir=pro_cfg.by_instru_pre_dir,
        db_name=db_struct["preprocess"]["db_name"],
        table=CSqlTable(cfg=db_struct["preprocess"]["table"]),
    ),
)
