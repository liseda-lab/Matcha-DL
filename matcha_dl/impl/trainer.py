import warnings
from typing import List, Optional

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from matcha_dl.core.contracts.trainer import EntityMapping, ITrainer


class MLPTrainer(ITrainer):

    def train(
        self,
        epochs: Optional[int] = 50,
        batch_size: Optional[int] = None,
        save_interval: Optional[int] = 5,
    ):

        warnings.filterwarnings("ignore", category=UserWarning)

        writer = SummaryWriter(self.logs_dir)

        while self.epoch <= epochs:
            self._model.train()
            _iter = 1

            with tqdm(self._load_data(kind="train", batch_size=batch_size), unit="batch") as tepoch:

                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    self._optimizer.zero_grad()
                    logits = self._model(data)
                    loss = self._loss(logits, target)
                    writer.add_scalar("Loss/train", loss, _iter)

                    loss.backward()
                    self._optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

                    _iter += 1

            if self.epoch % save_interval == 0:
                self.save_checkpoint()

            self._epoch += 1

        writer.flush()
        writer.close()

    def repair(self, **kwargs):

        # TODO add AML repair
        pass

    def predict(self, threshold: Optional[float] = 0.7, **kwargs) -> List[EntityMapping]:

        kind = "inference"

        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]

        # if supervised use model to calculate scores
        if self.dataset.reference is not None:

            data, _ = self._load_data(kind=kind)
            self._model.eval()
            with th.no_grad():
                logits = self._model(data)

            df["Scores"] = logits.cpu()

            return [
                EntityMapping(dp["SrcEntity"], dp["TgtEntity"], "=", dp["Scores"])
                for _, dp in df.iterrows()
                if dp["Scores"] >= threshold
            ]

        # if unsupervised use max score from matcha
        else:

            df["matcha"] = np.array(df["Features"].values.tolist()).max(axis=1)

            return [
                EntityMapping(dp["SrcEntity"], dp["TgtEntity"], "=", dp["matcha"])
                for _, dp in df.iterrows()
                if dp["matcha"] >= threshold
            ]

    def _load_data(
        self, kind: Optional[str] = "train", batch_size: Optional[int] = 1
    ) -> DataLoader:

        x = self.dataset.x(kind)
        y = self.dataset.y(kind)

        x = th.tensor(x, dtype=th.float32)
        x = x.to(self.device)

        y = th.tensor(y, dtype=th.float32)
        y = y.unsqueeze(1)
        y = y.to(self.device)

        if kind == "train":
            ds = TensorDataset(x, y)

            return DataLoader(ds, batch_size=batch_size, shuffle=True)

        return x, y
