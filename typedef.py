from dataclasses import dataclass
from husfort.qsqlite import CDbStruct

TInstruName = str


@dataclass(frozen=True)
class CCfgInstru:
    sectorL0: str
    sectorL1: str


TUniverse = dict[TInstruName, CCfgInstru]


@dataclass(frozen=True)
class CCfgAvlbUnvrs:
    win: int
    amount_threshold: float


@dataclass(frozen=True)
class CCfgConst:
    COST: float
    SECTORS: list[str]
    WIN: int
    LAG: int
    SHIFT: int
    RET_CLASS: str
    RET_NAMES: list[str]
    MAW: int


@dataclass(frozen=True)
class CProCfg:
    # --- shared
    calendar_path: str
    root_dir: str
    db_struct_path: str
    alternative_dir: str
    market_index_path: str
    by_instru_pos_dir: str
    by_instru_pre_dir: str

    # --- project
    project_root_dir: str
    available_dir: str
    market_dir: str

    # --- project parameters
    universe: TUniverse
    avlb_unvrs: CCfgAvlbUnvrs
    mkt_idxes: dict
    const: CCfgConst


@dataclass(frozen=True)
class CDbStructCfg:
    # --- shared database
    macro: CDbStruct
    forex: CDbStruct
    fmd: CDbStruct
    position: CDbStruct
    basis: CDbStruct
    stock: CDbStruct
    preprocess: CDbStruct

    # --- project database
    available: CDbStruct
    market: CDbStruct
