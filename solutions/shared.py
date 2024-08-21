def convert_mkt_idx(mkt_idx: str, prefix: str = "I") -> str:
    return f"{prefix}{mkt_idx.replace('.', '_')}"
