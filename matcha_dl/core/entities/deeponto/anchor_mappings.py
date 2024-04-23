
from deeponto import SavedObj, EntityMapping, EntityMappingList, OntoMappings

from typing import Optional
from matcha_dl.core.values import DEFAULT_REL, DEFAULT_DUP_STRATEGY
from matcha_dl.impl.deeponto.utils import read_table, sort_dict_by_values

from deeponto import SavedObj

from collections import defaultdict

import pandas as pd
import ast

class AnchorMapping(EntityMapping):
    """A special EntityMapping that serves as an "anchor" for its candidates"""

    def __init__(
        self,
        src_ent_iri: str,
        tgt_ent_iri: str,
        rel: str = DEFAULT_REL,
        score: float = 0.0,
        *cand_maps: EntityMapping,
    ):
        super().__init__(src_ent_iri, tgt_ent_iri, rel, score)
        self.candidates = EntityMappingList()
        for cand in cand_maps:
            self.add_candidate(cand)

    def __repr__(self):
        return (
            f"Anchor: EntityMapping({self.head} {self.rel} {self.tail}, {round(self.score, 6)})\n"
            + f"Candidates: {str(self.candidates)}"
        )

    def add_candidate(self, cand_map: EntityMapping):
        """Add candidate mappings whose relations and head entities are the
        same as the anchor mapping's.
        """
        if self.rel != cand_map.rel:
            raise ValueError(
                "Expect relation of candidate mapping to " + f"be {self.rel} but got {cand_map.rel}"
            )
        if self.head != cand_map.head:
            raise ValueError(
                "Candidate mapping does not have the same head entity as the anchor mapping."
            )
        self.candidates.append(cand_map)




class AnchoredOntoMappings(SavedObj):
    def __init__(
        self,
        flag: str,
        n_best: Optional[int],
        rel: str,
        dup_strategy: str = DEFAULT_DUP_STRATEGY,
        *anchor_mappings: AnchorMapping,
    ):
        """Store ranked (by score) mappings for each reference (head, tail) pairs:
        {
            ...
            ("anchor_head_ent_i", "anchor_tail_ent_i": {
            ...
            Sorted({
                ...
                "tail_ent_j": score(i, j) # NOTE: anchor_tail_ent_i is somewhere
                ...
                })
            ...
            }
            ...
        }
        NOTE: [anchor_head, anchor_tail] ensures uniqueness to a anchor mapping
        """
        self.flag = flag
        self.n_best = n_best
        self.rel = rel
        self.dup_strategy = dup_strategy
        # save mappings in disjoint partitions to prevent overriding keys (src entities)
        self.anchor2cands = defaultdict(dict)
        self.cand2anchors = defaultdict(list)
        self.add_many(*anchor_mappings)
        super().__init__(f"{self.flag}.anchored.maps")

    def __str__(self):
        self.info = {
            "flag": self.flag,
            "relation": self.rel,
            "n_best": self.n_best,
            "num_anchors": len(self.anchor2cands),
            "num_maps": len(self),
        }
        return super().report(**self.info)

    def __len__(self):
        """Total number of ranked mappings
        """
        return sum([len(map_dict) for map_dict in self.anchor2cands.values()])

    def save_instance(self, saved_path):
        """save the current instance locally
        """
        super().save_instance(saved_path)
        # also save a readable format of the ranked alignment set
        anchor2cand_json = {str(k): v for k, v in self.anchor2cands.items()}
        self.save_json(anchor2cand_json, saved_path + f"/{self.saved_name}.json")
        self.to_df().to_csv(saved_path + f"/{self.saved_name}.tsv", sep="\t", index=False)

    def add(self, anchor_map: AnchorMapping, allow_existed: bool = True):
        """Given an anchor mapping, add a new candidate mapping or add an existing
        candidate mapping to update mapping score (take average) while keeping the ranking
        """
        self.validate_anchor_mapping(anchor_map)
        existed_cands = self.existed_cands_for_anchor(anchor_map)
        new_cands = list(set(anchor_map.candidates) - set(existed_cands))

        # update new candidate mappings
        for cand_map in new_cands:
            self.anchor2cands[anchor_map.to_tuple()][cand_map.tail] = cand_map.score
            self.cand2anchors[cand_map.to_tuple()].append(anchor_map.tail)

        # address exisitng candidate mappings
        if existed_cands and not allow_existed:
            raise ValueError("Duplicate mappings not allowed ...")
        for cand_map in existed_cands:
            # average/kept_new/kept_old the mapping scores if already existed
            old_score = self.anchor2cands[anchor_map.to_tuple()][cand_map.tail]
            if self.dup_strategy == "average":
                new_score = (old_score + cand_map.score) / 2
            elif self.dup_strategy == "kept_new":
                new_score = cand_map.score
            else:  # for "kept_old"
                new_score = old_score
            self.anchor2cands[anchor_map.to_tuple()][cand_map.tail] = new_score
            self.cand2anchors[cand_map.to_tuple()].append(anchor_map.tail)
        # rank according to mapping scores and preserve n_best (if specified)
        self.anchor2cands[anchor_map.to_tuple()] = sort_dict_by_values(
            self.anchor2cands[anchor_map.to_tuple()], top_k=self.n_best
        )

    def add_many(self, *anchor_maps: AnchorMapping):
        """Add a list of anchor-cand mapping pairs while keeping the ranking
        """
        for am in anchor_maps:
            self.add(am)

    def fill_scored_maps(self, scored_onto_maps: OntoMappings):
        """Fill mapping score from scored onto mappings
        """
        assert self.flag == scored_onto_maps.flag
        num_valid = 0
        for src_ent_iri, v in scored_onto_maps.map_dict.items():
            for tgt_ent_iri, score in v.items():
                if self.cand2anchors[src_ent_iri, tgt_ent_iri]:
                    num_valid += 1
                    for anchor_tail in self.cand2anchors[src_ent_iri, tgt_ent_iri]:
                        self.anchor2cands[src_ent_iri, anchor_tail][tgt_ent_iri] = score
                    self.anchor2cands[src_ent_iri, anchor_tail] = sort_dict_by_values(
                        self.anchor2cands[src_ent_iri, anchor_tail], top_k=self.n_best
                    )
        print(
            f"{num_valid}/{len(scored_onto_maps)} of scored mappings are filled to corresponding anchors."
        )

    def unscored_cand_maps(self) -> OntoMappings:
        """Return all candidate mappings with no scores and anchors (so that duplicates will be merged)
        """
        unscored_cands = OntoMappings(self.flag, self.n_best, self.rel, self.dup_strategy)
        for cand_tup in self.cand2anchors.keys():
            cand_map = EntityMapping(cand_tup[0], cand_tup[1], self.rel, 0.0)
            unscored_cands.add(cand_map)
        return unscored_cands

    def to_df(self, with_scores=False):
        """Unravel the anchor mappings from dict to dataframe
        """
        anchor_map_df = pd.DataFrame(
            columns=["SrcEntity", "TgtEntity", "TgtCandidates", "CandScores"]
        )
        i = 0
        for ref_tup, v in self.anchor2cands.items():
            tgt_cands, cand_scores = zip(*v.items())
            anchor_map_df.loc[i] = [ref_tup[0], ref_tup[1], tgt_cands, cand_scores]
            i += 1
        if not with_scores:
            anchor_map_df = anchor_map_df.drop(columns=["CandScores"])
        return anchor_map_df

    def validate_anchor_mapping(self, am: AnchorMapping):
        if am.rel != self.rel:
            raise ValueError(f"Expect anchor mapping relation {self.rel}) but got {am.rel}.")

    def existed_cands_for_anchor(self, anchor_map: AnchorMapping):
        """Check if candidate mappings for an anchor already exist
        """
        existed_cands = EntityMappingList()
        for cand in anchor_map.candidates:
            if cand.tail in self.anchor2cands[anchor_map.to_tuple()].keys():
                existed_cands.append(cand)
        return existed_cands

    @classmethod
    def read_table_mappings(
        cls,
        table_mappings_path: str,
        flag: str = "src2tgt",
        n_best: Optional[int] = None,
        rel: str = DEFAULT_REL,
        dup_strategy: str = DEFAULT_DUP_STRATEGY,
    ):
        """Read mappings from csv/tsv files and preserve mappings with scores >= threshold
        """
        df = read_table(table_mappings_path)
        anchored_onto_mappings = cls(flag=flag, n_best=n_best, rel=rel, dup_strategy=dup_strategy)
        for _, dp in df.iterrows():
            anchor_map = AnchorMapping(dp["SrcEntity"], dp["TgtEntity"], rel, 1.0)
            tgt_cands = list(ast.literal_eval(dp["TgtCandidates"]))
            if "CandScores" in df.columns:
                cand_scores = ast.literal_eval(df["CandScores"])
            else:
                cand_scores = [0.0] * len(tgt_cands)
            for i in range(len(tgt_cands)):
                cand_map = EntityMapping(anchor_map.head, tgt_cands[i], rel, cand_scores[i])
                anchor_map.add_candidate(cand_map)
            anchored_onto_mappings.add(anchor_map)
        return anchored_onto_mappings