import os
import yaml
from typedef import TUniverse, CCfgInstru, CCfgAvlbUnvrs, CCfgConst, CCfgFeatSlc, CCfgTrn
from typedef import CCfgProj, CCfgDbStruct
from typedef import (
    CCfgFactors,
    CCfgFactorMTM,
    CCfgFactorSKEW,
    CCfgFactorRS,
    CCfgFactorBASIS,
    CCfgFactorTS,
    CCfgFactorS0BETA,
    CCfgFactorS1BETA,
    CCfgFactorCBETA,
    CCfgFactorIBETA,
    CCfgFactorPBETA,
    CCfgFactorCTP,
    CCfgFactorCTR,
    CCfgFactorCVP,
    CCfgFactorCVR,
    CCfgFactorCSP,
    CCfgFactorCSR,
    CCfgFactorNOI,
    CCfgFactorNDOI,
    CCfgFactorWNOI,
    CCfgFactorWNDOI,
    CCfgFactorAMP,
    CCfgFactorEXR,
    CCfgFactorSMT,
    CCfgFactorRWTC,
)
from typedef import TFactorsPool
from husfort.qsqlite import CDbStruct, CSqlTable

# ---------- project configuration ----------

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

universe: TUniverse = {k: CCfgInstru(**v) for k, v in _config["universe"].items()}

proj_cfg = CCfgProj(
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
    test_return_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["test_return_dir"]
    ),
    factors_by_instru_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["factors_by_instru_dir"]
    ),
    neutral_by_instru_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["neutral_by_instru_dir"]
    ),
    feature_selection_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["feature_selection_dir"]
    ),

    universe=universe,
    avlb_unvrs=CCfgAvlbUnvrs(**_config["available"]),
    mkt_idxes=_config["mkt_idxes"],
    const=CCfgConst(**_config["CONST"]),
    factors=_config["factors"],
    trn=CCfgTrn(**_config["trn"]),
    feat_slc=CCfgFeatSlc(**_config["feature_selection"]),
)

# ---------- databases structure ----------
with open(proj_cfg.db_struct_path, "r") as f:
    _db_struct = yaml.safe_load(f)

db_struct_cfg = CCfgDbStruct(
    # --- shared database
    macro=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
        db_name=_db_struct["macro"]["db_name"],
        table=CSqlTable(cfg=_db_struct["macro"]["table"]),
    ),
    forex=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
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

# --- factors ---
cfg_factors = CCfgFactors(
    MTM=CCfgFactorMTM(**proj_cfg.factors["MTM"]),
    SKEW=CCfgFactorSKEW(**proj_cfg.factors["SKEW"]),
    RS=CCfgFactorRS(**proj_cfg.factors["RS"]),
    BASIS=CCfgFactorBASIS(**proj_cfg.factors["BASIS"]),
    TS=CCfgFactorTS(**proj_cfg.factors["TS"]),
    S0BETA=CCfgFactorS0BETA(**proj_cfg.factors["S0BETA"]),
    S1BETA=CCfgFactorS1BETA(**proj_cfg.factors["S1BETA"]),
    CBETA=CCfgFactorCBETA(**proj_cfg.factors["CBETA"]),
    IBETA=CCfgFactorIBETA(**proj_cfg.factors["IBETA"]),
    PBETA=CCfgFactorPBETA(**proj_cfg.factors["PBETA"]),
    CTP=CCfgFactorCTP(**proj_cfg.factors["CTP"]),
    CTR=CCfgFactorCTR(**proj_cfg.factors["CTR"]),
    CVP=CCfgFactorCVP(**proj_cfg.factors["CVP"]),
    CVR=CCfgFactorCVR(**proj_cfg.factors["CVR"]),
    CSP=CCfgFactorCSP(**proj_cfg.factors["CSP"]),
    CSR=CCfgFactorCSR(**proj_cfg.factors["CSR"]),
    NOI=CCfgFactorNOI(**proj_cfg.factors["NOI"]),
    NDOI=CCfgFactorNDOI(**proj_cfg.factors["NDOI"]),
    WNOI=CCfgFactorWNOI(**proj_cfg.factors["WNOI"]),
    WNDOI=CCfgFactorWNDOI(**proj_cfg.factors["WNDOI"]),
    AMP=None,  # CCfgFactorAMP(**proj_cfg.factors["AMP"]),
    EXR=None,  # CCfgFactorEXR(**proj_cfg.factors["EXR"]),
    SMT=None,  # CCfgFactorSMT(**proj_cfg.factors["SMT"]),
    RWTC=None,  # CCfgFactorRWTC(**proj_cfg.factors["RWTC"]),
)

factors_pool_raw: TFactorsPool = []
factors_pool_neu: TFactorsPool = []
for cfg_factor in cfg_factors.values():
    factors_pool_raw.extend(cfg_factor.get_combs_raw(proj_cfg.factors_by_instru_dir))
    factors_pool_neu.extend(cfg_factor.get_combs_neu(proj_cfg.neutral_by_instru_dir))

if __name__ == "__main__":
    print(f"Size of universe = {len(universe)}")
    print("databases structures:")
    print(db_struct_cfg)
    print("project_configuration:")
    print(proj_cfg)
    print("factors pool raw")
    print(factors_pool_raw)
    print("factors pool neu")
    print(factors_pool_neu)
