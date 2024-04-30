# Adapted or copied from https://github.com/KRR-Oxford/DeepOnto

from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
from deeponto.utils import read_table
from . import SavedObj

from matcha_dl.core.values import DEFAULT_DUP_STRATEGY, DEFAULT_REL, DUP_STRATEGIES
from matcha_dl.impl.dp.utils import sort_dict_by_values


class EntityMapping:
    def __init__(
        self, src_ent_iri: str, tgt_ent_iri: str, rel: str = DEFAULT_REL, score: float = 0.0
    ):
        self.head = src_ent_iri
        self.tail = tgt_ent_iri
        self.rel = rel
        self.score = score

    def to_tuple(self):
        return (self.head, self.tail)

    def __repr__(self):
        return f"EntityMapping({self.head} {self.rel} {self.tail}, {round(self.score, 6)})"


class EntityMappingList(list):
    def append(self, em: EntityMapping):
        if isinstance(em, EntityMapping):
            super().append(em)
        else:
            raise TypeError("Only EntityMapping can be added to the list.")

    def topk(self, k: int):
        """Return top K scored mappings from the list"""
        return EntityMappingList(sorted(self, key=lambda x: x.score, reverse=True))[:k]

    def sorted(self):
        """Return the sorted entity mapping list"""
        return self.topk(k=len(self))

    def to_tuples(self):
        return [(em.head, em.tail) for em in self]

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        if type(item) is slice:
            return EntityMappingList(result)
        else:
            return result

    def __repr__(self):
        info = type(self).__name__ + "(\n"
        for em in self:
            info += "  " + str(em) + ",\n"
        info += ")"
        return info

    @classmethod
    def read_table_mappings(
        cls,
        table_mappings_path: str,
        threshold: Optional[float] = 0.0,
        rel: str = DEFAULT_REL,
        n_best: Optional[int] = None,
    ):
        """Read mappings from csv/tsv files and preserve mappings with scores >= threshold"""
        df = read_table(table_mappings_path)
        mappings = cls()
        for _, dp in df.iterrows():
            if dp["Score"] >= threshold:
                mappings.append(EntityMapping(dp["SrcEntity"], dp["TgtEntity"], rel, dp["Score"]))
        return mappings.topk(n_best)


class OntoMappings(SavedObj):
    def __init__(
        self,
        flag: str,
        n_best: Optional[int],
        rel: str,
        dup_strategy: str = DEFAULT_DUP_STRATEGY,
        *ent_mappings: EntityMapping,
    ):
        """Store ranked (by score) mappings for each head entity in Dict:
        {
            ...
            "head_ent_i": Sorted({
                ...
                "tail_ent_j": score(i, j)
                ...
                })
            ...
        }
        """
        self.flag = flag
        self.rel = rel
        self.n_best = n_best
        self.map_dict = defaultdict(dict)
        self.dup_strategy = dup_strategy
        assert self.dup_strategy in DUP_STRATEGIES
        self.add_many(*ent_mappings)
        super().__init__(f"{self.flag}.maps")

    def __str__(self):
        self.info = {
            "flag": self.flag,
            "relation": self.rel,
            "n_best": self.n_best,
            "num_heads": len(self.map_dict),
            "num_maps": len(self),
        }
        return super().report(**self.info)

    def __len__(self):
        """Total number of ranked mappings"""
        return sum([len(map_dict) for map_dict in self.map_dict.values()])

    def save_instance(self, saved_path):
        """save the current instance locally"""
        super().save_instance(saved_path)
        # also save in readable formats of the ranked alignment set
        self.save_json(self.map_dict, saved_path + f"/{self.saved_name}.json")
        self.to_df().to_csv(saved_path + f"/{self.saved_name}.tsv", sep="\t", index=False)

    def topks_for_ent(self, src_ent_iri: str, K: Optional[int] = None, threshold: float = 0.0):
        """Return ranked mappings for a particular entry (head) entity"""
        ls = EntityMappingList()
        for tgt_ent_iri, score in list(self.map_dict[src_ent_iri].items())[:K]:
            if score >= threshold:
                ls.append(EntityMapping(src_ent_iri, tgt_ent_iri, self.rel, score))
        return ls

    def topks(
        self,
        K: Optional[int] = 1,
        threshold: float = 0.0,
        as_tuples: bool = False,
        include_scores: bool = False,
    ):
        """Return the top ranked anchor mappings for each head entity with scores >= threshold,
        output mappings are transformed to tuples
        """
        # NOTE: when K = None, slicing automatically gives the whole length
        # i.e., ls[:None] == ls[:len(ls)]
        topks_for_all = dict()
        for src_ent_iri in self.map_dict.keys():
            topks_for_all[src_ent_iri] = self.topks_for_ent(src_ent_iri, K, threshold)
        if as_tuples:
            tps = []
            for v in topks_for_all.values():
                if not include_scores:
                    tps += [(em.head, em.tail) for em in v]
                else:
                    tps += [(em.head, em.tail, em.score) for em in v]
            return tps
        return topks_for_all

    def to_tuples(self, include_scores: bool = False) -> List[Tuple[str, str]]:
        """Unravel the mappings from dict to tuples"""
        return self.topks(K=None, as_tuples=True, include_scores=include_scores)

    def to_df(self):
        """Unravel the mappings from dict to dataframe"""
        triples = self.to_tuples(include_scores=True)
        map_df = pd.DataFrame(data=triples, columns=["SrcEntity", "TgtEntity", "Score"])
        return map_df

    def add(self, em: EntityMapping):
        """Add a new entity mapping or add an existing mapping to update mapping score (take average)
        while keeping the ranking
        """
        self.validate_mapping(em)
        # average the mapping scores if already existed
        if self.is_existed_mapping(em):
            old_score = self.map_dict[em.head][em.tail]
            if self.dup_strategy == "average":
                new_score = (old_score + em.score) / 2
            elif self.dup_strategy == "kept_new":
                new_score = em.score
            else:  # for "kept_old"
                new_score = old_score
            self.map_dict[em.head][em.tail] = new_score
        else:
            self.map_dict[em.head][em.tail] = em.score
        # rank according to mapping scores and preserve n_best (if specified)
        self.map_dict[em.head] = sort_dict_by_values(self.map_dict[em.head], top_k=self.n_best)

    def add_many(self, *ems: EntityMapping):
        """Add a list of new mappings while keeping the ranking"""
        for em in ems:
            self.add(em)

    def validate_mapping(self, em: EntityMapping):
        if em.relation != self.rel:
            raise ValueError(f"expected mapping relation: {self.rel}) but received {em.relation}")

    def is_existed_mapping(self, em: EntityMapping):
        return em.tail in self.map_dict[em.head].keys()

    @classmethod
    def read_table_mappings(
        cls,
        table_mappings_path: str,
        flag: str = "src2tgt",
        n_best: Optional[int] = None,
        rel: str = DEFAULT_REL,
        dup_strategy: str = DEFAULT_DUP_STRATEGY,
    ):
        """Read mappings from csv/tsv files and preserve mappings with scores >= threshold"""
        df = read_table(table_mappings_path)
        onto_mappings = cls(flag=flag, n_best=n_best, rel=rel, dup_strategy=dup_strategy)
        for _, dp in df.iterrows():
            score = 0.0
            if "Score" in df.columns:
                score = dp["Score"]
            onto_mappings.add(EntityMapping(dp["SrcEntity"], dp["TgtEntity"], rel, score))
        return onto_mappings
