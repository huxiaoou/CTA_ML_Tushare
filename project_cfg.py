import os
import yaml
from typedef import TUniverse, CCfgInstru, CCfgAvlbUnvrs, CCfgConst
from typedef import CProCfg, CDbStructCfg
from husfort.qsqlite import CDbStruct, CSqlTable

# ---------- project configuration ----------

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

universe: TUniverse = {k: CCfgInstru(**v) for k, v in _config["universe"].items()}

proj_cfg = CProCfg(
    # --- shared
    calendar_path=_config["path"]["calendar_path"],
    root_dir=_config["path"]["root_dir"],
    db_struct_path=_config["path"]["db_struct_path"],
    alternative_dir=_config["path"]["alternative_dir"],
    market_index_path=_config["path"]["market_index_path"],
    by_instru_pos_dir=_config["path"]["by_instru_pos_dir"],
    by_instru_pre_dir=_config["path"]["by_instru_pre_dir"],

    # --- project
    project_root_dir=_config["path"]["project_root_dir"],
    available_dir=os.path.join(_config["path"]["project_root_dir"], _config["path"]["available_dir"]),  # type:ignore
    market_dir=os.path.join(_config["path"]["project_root_dir"], _config["path"]["market_dir"]),  # type:ignore

    universe=universe,
    avlb_unvrs=CCfgAvlbUnvrs(**_config["available"]),
    mkt_idxes=_config["mkt_idxes"],
    const=CCfgConst(**_config["CONST"]),
)

# ---------- databases structure ----------
with open(proj_cfg.db_struct_path, "r") as f:
    _db_struct = yaml.safe_load(f)

db_struct_cfg = CDbStructCfg(
    # --- shared database
    macro=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["macro"]["db_name"],
        table=CSqlTable(cfg=_db_struct["macro"]["table"]),
    ),
    forex=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["forex"]["db_name"],
        table=CSqlTable(cfg=_db_struct["forex"]["table"]),
    ),
    fmd=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["fmd"]["db_name"],
        table=CSqlTable(cfg=_db_struct["fmd"]["table"]),
    ),
    position=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["position"]["db_name"],
        table=CSqlTable(cfg=_db_struct["position"]["table"]),
    ),
    basis=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["basis"]["db_name"],
        table=CSqlTable(cfg=_db_struct["basis"]["table"]),
    ),
    stock=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["stock"]["db_name"],
        table=CSqlTable(cfg=_db_struct["stock"]["table"]),
    ),
    preprocess=CDbStruct(
        db_save_dir=proj_cfg.by_instru_pre_dir,
        db_name=_db_struct["preprocess"]["db_name"],
        table=CSqlTable(cfg=_db_struct["preprocess"]["table"]),
    ),

    # --- project database
    available=CDbStruct(
        db_save_dir=proj_cfg.available_dir,
        db_name=_config["db_struct"]["available"]["db_name"],
        table=CSqlTable(cfg=_config["db_struct"]["available"]["table"]),
    ),
    market=CDbStruct(
        db_save_dir=proj_cfg.market_dir,
        db_name=_config["db_struct"]["market"]["db_name"],
        table=CSqlTable(cfg=_config["db_struct"]["market"]["table"]),
    ),
)

if __name__ == "__main__":
    print(f"Size of universe = {len(universe)}")
    print("databases structures:")
    print(db_struct_cfg)
    print("project_configuration:")
    print(proj_cfg)
