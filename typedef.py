from dataclasses import dataclass
from husfort.qsqlite import CDbStruct

TInstruName = str


@dataclass(frozen=True)
class CInstruCfg:
    sectorL0: str
    sectorL1: str


TUniverse = dict[TInstruName, CInstruCfg]


@dataclass(frozen=True)
class CProCfg:
    calendar_path: str
    root_dir: str
    daily_data_root_dir: str
    db_struct_path: str
    alternative_dir: str
    by_instru_pos_dir: str
    by_instru_pre_dir: str
    universe: TUniverse


@dataclass(frozen=True)
class CDbStructCfg:
    macro: CDbStruct
    forex: CDbStruct
    fmd: CDbStruct
    position: CDbStruct
    basis: CDbStruct
    stock: CDbStruct
    preprocess: CDbStruct
