from .dataset_loader import ImageDataset
from .ltcc_noneID import LTCC_noneID
from .ltcc_orig import LTCC_Orig
from .celebreid_noneID import CELEBREID_noneID
from .celebreid_orig import CELEBREID_Orig
from .prcc_noneID import PRCC_noneID
from .prcc_orig import PRCC_Orig
from .prcc_orig_shu2021 import PRCC_Orig_Shu2021
from .nkup_noneID import NKUP_noneID
from .nkup_orig import NKUP_Orig

__factory = {
    'ltcc_noneID': LTCC_noneID,
    'ltcc_orig': LTCC_Orig,
    'celebreid_noneID': CELEBREID_noneID,
    'celebreid_orig': CELEBREID_Orig,
    'prcc_noneID': PRCC_noneID,
    'prcc_orig': PRCC_Orig,
    'prcc_orig_shu2021': PRCC_Orig_Shu2021,
    'nkup_noneID': NKUP_noneID,
    'nkup_orig': NKUP_Orig
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
