"""NewsTorch model exports, loaded on demand to isolate optional dependencies."""
from importlib import import_module

__all__ = [
    'CenNewsRec', 'CNE_SUE', 'CNRCL', 'CPRS', 'DKN', 'EENR', 'FIM', 'IPNR',
    'LKPNR', 'LSTUR', 'MINS', 'MMRec', 'NAML', 'NPA', 'NRMS', 'PLM_NR',
    'ONCE_DIRE_LLAMA1_NAML', 'ONCE_DIRE_QWEN3_NAML', 'PNR_LLM', 'S2LENR',
    'SentiDebias', 'SentiRec', 'TANR', 'TCCM',
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{name}'), name)
    globals()[name] = value
    return value
