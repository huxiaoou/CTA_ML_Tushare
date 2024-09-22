import itertools as ittl
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
class CCfgTrn:
    wins: list[int]


@dataclass(frozen=True)
class CCfgFeatSlc:
    mut_info_threshold: float
    min_feats: int


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
class CCfgProj:
    # --- shared
    calendar_path: str
    root_dir: str
    db_struct_path: str
    alternative_dir: str
    market_index_path: str
    by_instru_pos_dir: str
    by_instru_pre_dir: str
    by_instru_min_dir: str

    # --- project
    project_root_dir: str
    available_dir: str
    market_dir: str
    test_return_dir: str
    factors_by_instru_dir: str
    neutral_by_instru_dir: str
    feature_selection_dir: str
    mclrn_dir: str
    mclrn_cfg_file: str
    mclrn_mdl_dir: str
    mclrn_prd_dir: str
    signals_dir: str
    signals_mdl_dir: str
    signals_pfo_dir: str
    simu_dir: str
    simu_mdl_dir: str
    simu_pfo_dir: str
    eval_dir: str
    eval_mdl_dir: str
    eval_pfo_dir: str

    # --- project parameters
    universe: TUniverse
    avlb_unvrs: CCfgAvlbUnvrs
    mkt_idxes: dict
    const: CCfgConst
    factors: dict
    trn: CCfgTrn
    feat_slc: CCfgFeatSlc
    mclrn: dict[str, dict]
    portfolios: dict[str, dict]


@dataclass(frozen=True)
class CCfgDbStruct:
    # --- shared database
    macro: CDbStruct
    forex: CDbStruct
    fmd: CDbStruct
    position: CDbStruct
    basis: CDbStruct
    stock: CDbStruct
    preprocess: CDbStruct
    minute_bar: CDbStruct

    # --- project database
    available: CDbStruct
    market: CDbStruct


"""
---------------------------------------
Part II: Classes and types for factor configuration
---------------------------------------
"""

TFactorClass = str
TFactorName = str
TFactorNames = list[TFactorName]
TFactorClassAndNames = tuple[TFactorClass, TFactorNames]
TFactorComb = tuple[TFactorClass, TFactorNames, str]  # str is for subdirectory
TFactor = tuple[TFactorClass, TFactorName]
TFactorsPool = list[TFactorComb]


@dataclass(frozen=True)
class CCfgFactor:
    @property
    def factor_class(self) -> TFactorClass:
        raise NotImplementedError

    @property
    def factor_names(self) -> TFactorNames:
        raise NotImplementedError

    @property
    def factor_names_neu(self) -> TFactorNames:
        return [_.replace("RAW", "NEU") for _ in self.factor_names]

    def get_raw_class_and_names(self) -> TFactorClassAndNames:
        return self.factor_class, self.factor_names

    def get_neu_class_and_names(self) -> TFactorClassAndNames:
        neu_names = [_.replace("RAW", "NEU") for _ in self.factor_names]
        return self.factor_class, neu_names

    def get_combs_raw(self, sub_dir: str) -> TFactorsPool:
        factor_class, factor_names = self.get_raw_class_and_names()
        return [(factor_class, factor_names, sub_dir)]

    def get_combs_neu(self, sub_dir: str) -> TFactorsPool:
        factor_class, factor_names = self.get_neu_class_and_names()
        return [(factor_class, factor_names, sub_dir)]

    def get_combs(self, sub_dir: str) -> TFactorsPool:
        return self.get_combs_raw(sub_dir) + self.get_combs_neu(sub_dir)


# cfg for factors
@dataclass(frozen=True)
class CCfgFactorMTM(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "MTM"

    @property
    def factor_names(self) -> TFactorNames:
        return [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]


@dataclass(frozen=True)
class CCfgFactorSKEW(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "SKEW"

    @property
    def factor_names(self) -> TFactorNames:
        return [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]


@dataclass(frozen=True)
class CCfgFactorRS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "RS"

    @property
    def factor_names(self) -> TFactorNames:
        rspa: TFactorNames = [f"{self.factor_class}PA{w:03d}_RAW" for w in self.wins]
        rsla: TFactorNames = [f"{self.factor_class}LA{w:03d}_RAW" for w in self.wins]
        return rspa + rsla


@dataclass(frozen=True)
class CCfgFactorBASIS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "BASIS"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}D{w:03d}_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorTS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "TS"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}D{w:03d}_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorS0BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "S0BETA"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}{self.wins[0]:03d}D{w:03d}_RAW" for w in self.wins[1:]]
        # n2: TFactorNames = [f"{self.factor_class}{w:03d}RES_RAW" for w in self.wins]
        # n3: TFactorNames = [f"{self.factor_class}{w:03d}RESSTD_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorS1BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "S1BETA"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}{self.wins[0]:03d}D{w:03d}_RAW" for w in self.wins[1:]]
        # n2: TFactorNames = [f"{self.factor_class}{w:03d}RES_RAW" for w in self.wins]
        # n3: TFactorNames = [f"{self.factor_class}{w:03d}RESSTD_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorCBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "CBETA"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}{self.wins[0]:03d}D{w:03d}_RAW" for w in self.wins[1:]]
        # n2: TFactorNames = [f"{self.factor_class}{w:03d}RES_RAW" for w in self.wins]
        # n3: TFactorNames = [f"{self.factor_class}{w:03d}RESSTD_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorIBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "IBETA"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}{self.wins[0]:03d}D{w:03d}_RAW" for w in self.wins[1:]]
        # n2: TFactorNames = [f"{self.factor_class}{w:03d}RES_RAW" for w in self.wins]
        # n3: TFactorNames = [f"{self.factor_class}{w:03d}RESSTD_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorPBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "PBETA"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"{self.factor_class}{self.wins[0]:03d}D{w:03d}_RAW" for w in self.wins[1:]]
        # n2: TFactorNames = [f"{self.factor_class}{w:03d}RES_RAW" for w in self.wins]
        # n3: TFactorNames = [f"{self.factor_class}{w:03d}RESSTD_RAW" for w in self.wins]
        return n0 + n1


@dataclass(frozen=True)
class CCfgFactorCTP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CTP"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorCTR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CTR"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorCVP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CVP"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorCVR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CVR"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorCSP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CSP"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorCSR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "CSR"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{int(t * 10):02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "NOI"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{t:02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "NDOI"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{t:02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorWNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "WNOI"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{t:02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorWNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "WNDOI"

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [f"{self.factor_class}{w:03d}T{t:02d}_RAW" for w, t in ittl.product(self.wins, self.tops)]
        return n0


@dataclass(frozen=True)
class CCfgFactorAMP(CCfgFactor):
    wins: list[int]
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "AMP"

    @property
    def factor_names(self) -> TFactorNames:
        nh: TFactorNames = [f"{self.factor_class}{w:03d}T{int(l * 10):02d}H_RAW" for w, l in
                            ittl.product(self.wins, self.lbds)]
        nl: TFactorNames = [f"{self.factor_class}{w:03d}T{int(l * 10):02d}L_RAW" for w, l in
                            ittl.product(self.wins, self.lbds)]
        nd: TFactorNames = [f"{self.factor_class}{w:03d}T{int(l * 10):02d}D_RAW" for w, l in
                            ittl.product(self.wins, self.lbds)]
        return nh + nl + nd


@dataclass(frozen=True)
class CCfgFactorEXR(CCfgFactor):
    wins: list[int]
    dfts: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "EXR"

    @property
    def factor_names(self) -> TFactorNames:
        n0: TFactorNames = [f"{self.factor_class}{w:03d}_RAW" for w in self.wins]
        n1: TFactorNames = [f"DXR{w:03d}D{d:02d}_RAW" for w, d in ittl.product(self.wins, self.dfts)]
        n2: TFactorNames = [f"AXR{w:03d}D{d:02d}_RAW" for w, d in ittl.product(self.wins, self.dfts)]
        return n0 + n1 + n2


@dataclass(frozen=True)
class CCfgFactorSMT(CCfgFactor):
    wins: list[int]
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return "SMT"

    @property
    def factor_names(self) -> TFactorNames:
        n_prc: TFactorNames = [f"{self.factor_class}{w:03d}T{int(l * 10):02d}P_RAW" for w, l in
                               ittl.product(self.wins, self.lbds)]
        n_ret: TFactorNames = [f"{self.factor_class}{w:03d}T{int(l * 10):02d}R_RAW" for w, l in
                               ittl.product(self.wins, self.lbds)]
        return n_prc + n_ret


@dataclass(frozen=True)
class CCfgFactorRWTC(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return "RWTC"

    @property
    def factor_names(self) -> TFactorNames:
        nu = [f"{self.factor_class}{w:03d}U_RAW" for w in self.wins]
        nd = [f"{self.factor_class}{w:03d}D_RAW" for w in self.wins]
        nt = [f"{self.factor_class}{w:03d}T_RAW" for w in self.wins]
        nv = [f"{self.factor_class}{w:03d}V_RAW" for w in self.wins]
        return nu + nd + nt + nv


@dataclass(frozen=True)
class CCfgFactorTA(CCfgFactor):
    macd: tuple[int, int, int]
    sar: tuple[float, float]

    @property
    def factor_class(self) -> TFactorClass:
        return "TA"

    @property
    def name_macd(self) -> TFactorName:
        fast, slow, diff = self.macd
        return f"{self.factor_class}MACDF{fast}S{slow}D{diff}_RAW"

    @property
    def name_sar(self) -> TFactorName:
        acceleration, maximum = self.sar
        return f"{self.factor_class}SARA{int(acceleration * 100):02d}M{int(maximum * 100):02d}_RAW"

    @property
    def factor_names(self) -> TFactorNames:
        names_ta = [self.name_macd, self.name_sar]
        return names_ta


@dataclass(frozen=True)
class CCfgFactors:
    MTM: CCfgFactorMTM | None
    SKEW: CCfgFactorSKEW | None
    RS: CCfgFactorRS | None
    BASIS: CCfgFactorBASIS | None
    TS: CCfgFactorTS | None
    S0BETA: CCfgFactorS0BETA | None
    S1BETA: CCfgFactorS1BETA | None
    CBETA: CCfgFactorCBETA | None
    IBETA: CCfgFactorIBETA | None
    PBETA: CCfgFactorPBETA | None
    CTP: CCfgFactorCTP | None
    CTR: CCfgFactorCTR | None
    CVP: CCfgFactorCVP | None
    CVR: CCfgFactorCVR | None
    CSP: CCfgFactorCSP | None
    CSR: CCfgFactorCSR | None
    NOI: CCfgFactorNOI | None
    NDOI: CCfgFactorNDOI | None
    WNOI: CCfgFactorWNOI | None
    WNDOI: CCfgFactorWNDOI | None
    AMP: CCfgFactorAMP | None
    EXR: CCfgFactorEXR | None
    SMT: CCfgFactorSMT | None
    RWTC: CCfgFactorRWTC | None
    TA: CCfgFactorTA | None

    def values(self) -> list[CCfgFactor]:
        res = []
        for _, v in vars(self).items():
            if v is not None:
                res.append(v)
        return res


# --- feature selection ---
TReturnClass = str
TReturnName = str
TReturnNames = list[TReturnName]
TReturnComb = tuple[TReturnClass, TReturnName, str]
TReturn = tuple[TReturnClass, TReturnName]


@dataclass(frozen=True)
class CRet:
    ret_class: TReturnClass
    ret_name: TReturnName
    shift: int

    @property
    def desc(self) -> str:
        return f"{self.ret_class}.{self.ret_name}"

    @property
    def win(self) -> int:
        return int(self.ret_class[0:3])  # "001L1"-> 1


@dataclass(frozen=True)
class CTestFtSlc:
    trn_win: int
    sector: str
    ret: CRet

    @property
    def save_id(self) -> str:
        return f"{self.ret.ret_name}-W{self.trn_win:03d}"


# --- machine learning models ---
TMdlDesc = str


@dataclass(frozen=True)
class CModel:
    model_type: str
    model_args: dict

    @property
    def desc(self) -> TMdlDesc:
        return f"{self.model_type}"


TUniqueId = str
TWinTxt = str
TSector = str


@dataclass(frozen=True)
class CTestMdl:
    unique_Id: TUniqueId
    trn_win: int
    sector: TSector
    ret: CRet
    model: CModel

    @property
    def tw(self) -> TWinTxt:
        return f"W{self.trn_win:03d}"

    @property
    def layers(self) -> list[str]:
        return [
            self.ret.ret_class,  # 001L1-NEU
            self.tw,  # W060
            self.model.desc,  # Ridge
            self.sector,  # AGR
            self.unique_Id,  # M0005
            self.ret.ret_name,  # CloseRtn001L1
        ]

    @property
    def save_tag_mdl(self) -> str:
        return ".".join(self.layers)


# --- simulations ---
@dataclass(frozen=True)
class CSimArgs:
    sim_id: str
    tgt_ret: CRet
    db_struct_sig: CDbStruct
    db_struct_ret: CDbStruct
    cost: float


TPid = str
TTarget = str
TWeights = dict[TUniqueId, float]


@dataclass(frozen=True)
class CPortfolioArgs:
    pid: TPid
    target: TTarget
    weights: TWeights
    portfolio_sim_args: dict[str, CSimArgs]


TSimArgsPriKey = tuple[TReturnClass, TWinTxt, TMdlDesc, TSector]  # ret_class, trn_win, model_desc, sector
TSimArgsSecKey = tuple[TSector, TUniqueId]  # sector, unique_id
TSimArgsGrp = dict[TSimArgsPriKey, dict[TSimArgsSecKey, CSimArgs]]

TSimArgsPriKeyBySec = TSector  # sector
TSimArgsSecKeyBySec = tuple[
    TReturnClass, TWinTxt, TMdlDesc, TReturnName, TUniqueId  # (ret_class, trn_win, model_desc, ret_name, unique_id)
]
TSimArgsGrpBySec = dict[TSimArgsPriKeyBySec, dict[TSimArgsSecKeyBySec, CSimArgs]]
