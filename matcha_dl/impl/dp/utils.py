# Adapted or copied from https://github.com/KRR-Oxford/DeepOnto

from typing import Optional

from deeponto.align.mapping import EntityMapping


def sort_dict_by_values(dic: dict, desc: bool = True, top_k: Optional[int] = None):
    """Return a sorted dict by values with top k reserved"""
    top_k = len(dic) if not top_k else top_k
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:top_k])


def fill_anchored_scores(ref_anchored_maps, pred_maps):
    """Fill scores of the anchored reference mappings with the scores of the predicted mappings."""

    pred_maps_tuples = EntityMapping.as_tuples(pred_maps, with_score=True)

    pred_maps_dict = {}
    for source, tgt, score in pred_maps_tuples:
        if not source in pred_maps_dict:
            pred_maps_dict[source] = {}
        pred_maps_dict[source][tgt] = score

    results = []
    for src_ref_class, tgt_ref_class, tgt_cands in ref_anchored_maps:
        tgt_cands = eval(tgt_cands)
        scored_cands = []
        for tgt_cand in tgt_cands:
            try:
                scored_cands.append((tgt_cand, pred_maps_dict[src_ref_class][tgt_cand]))

            except KeyError:
                scored_cands.append((tgt_cand, 0.0))

        results.append((src_ref_class, tgt_ref_class, scored_cands))
    return results
