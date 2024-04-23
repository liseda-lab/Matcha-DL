# Adapted or copied from https://github.com/KRR-Oxford/DeepOnto

from typing import Optional
import pandas as pd


def sort_dict_by_values(dic: dict, desc: bool = True, top_k: Optional[int] = None):
    """Return a sorted dict by values with top k reserved
    """
    top_k = len(dic) if not top_k else top_k
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:top_k])

def read_table(file_path: str):
    """Read tsv file as pandas dataframe without treating "null" as empty string.
    """
    sep = "\t" if file_path.endswith(".tsv") else ","
    na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})
    
    return pd.read_csv(file_path, sep=sep, na_values=na_vals, keep_default_na=False)