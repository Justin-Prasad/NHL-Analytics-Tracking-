"""
models/sequence_model.py

Bidirectional LSTM + Attention for possession sequence classification.

Given a sequence of hockey events (passes, entries, shots, etc.) with
spatial coordinates and game context, predicts whether the possession
will produce a scoring chance within 10 seconds.

Architecture:
    Input: (batch, seq_len, input_size)
    → Bidirectional LSTM (2 layers, hidden=128)
    → Attention pooling across timesteps
    → Dense head → sigmoid

The attention weights expose which events the model focuses on —
useful for communicating to coaches what makes a possession dangerous.

Usage:
    from models.sequence_model import SequenceLSTM, SequenceTrainer

    model = SequenceLSTM(input_size=16)
    trainer = SequenceTrainer(model)
    trainer.fit(train_sequences, val_sequences)
    probs = trainer.predict(test_sequences)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import LSTM_PARAMS, SEQUENCE_MAX_LEN

logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Wraps possession sequences for DataLoader.
    Each item: padded event tensor + label.
    """

    def __init__(self, sequences: list[dict], max_len: int = SEQUENCE_MAX_LEN):
        self.max_len = max_len
        self.data = []

        for seq in sequences:
            events = seq["events"]
            label = seq["label"]

            # Build feature matrix per event
            # Features: [event_type_onehot(13), x_norm, y_norm] = 15 dims
            # + time_norm = 16 dims total
            feature_matrix = []
            for ev in events[-max_len:]:
                one_hot = [0.0] * 13
                one_hot[min(ev["event_type"], 12)] = 1.0
                x_norm = (ev["x_coord"] + 100) / 200.0
                y_norm = (ev["y_coord"] + 42.5) / 85.0
                t_norm = ev["time"] / 1200.0
                feature_matrix.append(one_hot + [x_norm, y_norm, t_norm])

            # Left-pad to max_len
            pad_len = max_len - len(feature_matrix)
            padded = [[0.0] * LSTM_PARAMS["input_size"]] * pad_len + feature_matrix
            tensor = torch.tensor(padded, dtype=torch.float32)
            mask = torch.tensor(
                [False] * pad_len + [True] * len(feature_matrix), dtype=torch.bool
            )
            self.data.append((tensor, mask, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Attention Module ───────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over LSTM outputs.
    Returns context vector + attention weights.
    Weights are interpretable: high weight = important event in sequence.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        lstm_out: (batch, seq_len, hidden_size)
        mask: (batch, seq_len) bool, True = real token
        Returns: context (batch, hidden_size), weights (batch, seq_len)
        """
        scores = self.score(lstm_out).squeeze(-1)       # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)          # (batch, seq_len)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, hidden)
        return context, weights


# ── Model ──────────────────────────────────────────────────────────────────────

class SequenceLSTM(nn.Module):
    """
    Bidirectional LSTM + attention for possession sequence scoring.

    Forward pass returns (logits, attention_weights).
    attention_weights can be returned to stakeholders: which events
    drove the prediction.
    """

    def __init__(self,
                 input_size: int = LSTM_PARAMS["input_size"],
                 hidden_size: int = LSTM_PARAMS["hidden_size"],
                 num_layers: int = LSTM_PARAMS["num_layers"],
                 dropout: float = LSTM_PARAMS["dropout"],
                 bidirectional: bool = LSTM_PARAMS["bidirectional"]):
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attention = AdditiveAttention(hidden_size * directions)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        mask: (batch, seq_len) bool
        Returns: logits (batch,), attn_weights (batch, seq_len)
        """
        lstm_out, _ = self.lstm(x)              # (batch, seq, hidden*dir)
        context, weights = self.attention(lstm_out, mask)
        context = self.dropout(context)
        logits = self.head(context).squeeze(-1)  # (batch,)
        return logits, weights


# ── Trainer ────────────────────────────────────────────────────────────────────

class SequenceTrainer:
    """
    Handles training loop, early stopping, and evaluation for SequenceLSTM.
    """

    def __init__(self, model: SequenceLSTM, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"SequenceTrainer using device: {self.device}")

    def fit(self, train_seqs: list[dict], val_seqs: list[dict]) -> "SequenceTrainer":
        params = LSTM_PARAMS
        train_ds = SequenceDataset(train_seqs)
        val_ds = SequenceDataset(val_seqs)

        train_dl = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=params["batch_size"] * 2,
                            shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(params["epochs"]):
            # Train
            self.model.train()
            train_losses = []
            for x, mask, labels in train_dl:
                x = x.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(x, mask)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            val_loss, val_auc = self._evaluate(val_dl, criterion)
            train_loss = np.mean(train_losses)
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            logger.info(f"Epoch {epoch+1:3d} | "
                        f"Train {train_loss:.4f} | "
                        f"Val {val_loss:.4f} | "
                        f"AUC {val_auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.clone() for k, v in
                                    self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= params["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        self.model.load_state_dict(self._best_state)
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return self

    def predict(self, sequences: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (probabilities, attention_weights) arrays.
        attention_weights shape: (n_sequences, max_len) — interpretable per sequence.
        """
        ds = SequenceDataset(sequences)
        dl = DataLoader(ds, batch_size=256, shuffle=False)

        probs_list, weights_list = [], []
        self.model.eval()
        with torch.no_grad():
            for x, mask, _ in dl:
                x, mask = x.to(self.device), mask.to(self.device)
                logits, weights = self.model(x, mask)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                weights_list.append(weights.cpu().numpy())

        return np.concatenate(probs_list), np.concatenate(weights_list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "history": self.history,
        }, path)
        logger.info(f"Saved sequence model to {path}")

    @classmethod
    def load(cls, path: str | Path, model: SequenceLSTM) -> "SequenceTrainer":
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        trainer = cls(model)
        trainer.history = ckpt.get("history", {})
        return trainer

    # ── Internal ───────────────────────────────────────────────────────────────

    def _evaluate(self, dl: DataLoader, criterion) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        losses, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for x, mask, labels in dl:
                x, mask, labels = (t.to(self.device) for t in (x, mask, labels))
                logits, _ = self.model(x, mask)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = np.mean(losses)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5

        return val_loss, val_auc
