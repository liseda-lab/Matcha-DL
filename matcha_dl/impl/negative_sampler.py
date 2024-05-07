from matcha_dl.core.contracts.negative_sampler import INegativeSampler, List


class RandomNegativeSampler(INegativeSampler):

    def sample(self, sources: List, targets: List) -> List[List[str]]:

        cands = {
            src: [cand for cand in targets if cand != trg] for src, trg in zip(sources, targets)
        }

        if len(targets) < self.n_samples + 1:
            return [[source, candidate, 0.0] for source in sources for candidate in cands[source]]

        return [
            [source, candidate, 0.0]
            for source in sources
            for candidate in self.random.choice(cands[source], self.n_samples, replace=False)
        ]
