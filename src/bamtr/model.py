"""BAMTR: shared encoding, historical retrieval and missing-point decoding.

This is a new implementation of the supplied manuscript, not recovered author
code or pretrained weights. Underspecified choices are documented explicitly:
attention widths equal the bidirectional encoder width; the two attention
modules have independent projections; predicted stay probabilities pass through
the shared mobility embedding; and dropout is also applied at recurrent inputs
and outputs so that it remains effective with one recurrent layer.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import DEFAULTS


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Normalize valid entries; an entirely absent memory has zero weights."""
    if scores.shape[-1] == 0:
        return torch.zeros_like(scores)
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.softmax(masked, dim=-1).masked_fill(~mask, 0.0)


class BAMTR(nn.Module):
    """Recover the chronologically ordered missing points in a padded batch.

    ``encoder_hidden_size`` is the width of EACH BiGRU direction. Decoder width
    is independent and is aligned by learned projections. Target coordinates
    are read only in training when coordinate teacher forcing is enabled;
    target stay labels are never read by the model. Known target timestamps
    determine output order/length. By default their numeric values are not
    decoder features, matching Eq. (22). ``query_time_conditioning=True`` adds
    the current target day fraction as one decoder input: an explicit extension.

    No-history samples use current evidence alone. Disabling the context adaptor
    uses parameter-free current-plus-history fusion, an ablation convention.
    Forecasts at padded target positions are zero and must be masked in losses.
    """

    def __init__(self, model_config: dict[str, Any], environment_dim: int = 5):
        super().__init__()
        cfg = {**DEFAULTS["model"], **model_config}
        self.config = cfg
        self.encoder_hidden_size = int(cfg["encoder_hidden_size"])
        self.encoder_width = 2 * self.encoder_hidden_size
        self.decoder_hidden_size = int(cfg["decoder_hidden_size"])
        self.num_layers = int(cfg["num_layers"])
        self.use_history = bool(cfg["use_history"])
        self.use_context_adaptor = bool(cfg["use_context_adaptor"])
        self.use_mobility = bool(cfg["use_mobility"])
        self.use_environment = bool(cfg["use_environment"])
        self.use_local_attention = bool(cfg["use_local_attention"])
        self.query_time_conditioning = bool(cfg["query_time_conditioning"])
        self.environment_dim = int(environment_dim)
        probability = float(cfg["dropout"])
        semantic_dim = int(cfg["semantic_embedding_dim"]) if self.use_mobility else 0
        environmental_dim = int(cfg["environment_embedding_dim"]) if self.use_environment else 0
        head_dim = int(cfg["coord_head_dim"])
        if min(self.encoder_hidden_size, self.decoder_hidden_size, self.num_layers, head_dim) < 1:
            raise ValueError("Hidden sizes, recurrent layers and head width must be positive.")
        if not 0 <= probability < 1:
            raise ValueError("Dropout must lie in [0, 1).")
        if self.use_mobility and semantic_dim < 1:
            raise ValueError("The mobility embedding width must be positive.")
        if self.use_environment and min(environmental_dim, self.environment_dim) < 1:
            raise ValueError("Environment input and embedding widths must be positive.")

        e, d = self.encoder_width, self.decoder_hidden_size
        self.mobility_embedding = nn.Sequential(nn.Linear(1, semantic_dim), nn.Tanh()) if self.use_mobility else None
        self.environment_embedding = nn.Sequential(nn.Linear(self.environment_dim, environmental_dim), nn.Tanh()) if self.use_environment else None
        recurrent_dropout = probability if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(3 + semantic_dim, self.encoder_hidden_size,
                              num_layers=self.num_layers, batch_first=True,
                              bidirectional=True, dropout=recurrent_dropout)
        self.encoder_input_dropout = nn.Dropout(probability)
        self.encoder_output_dropout = nn.Dropout(probability)
        self.pool_projection = nn.Linear(e, e)
        self.pool_context = nn.Parameter(torch.empty(e))
        nn.init.normal_(self.pool_context, std=1 / math.sqrt(e))

        self.history_query = nn.Linear(e, e, bias=False) if self.use_history else None
        self.history_key = nn.Linear(e, e, bias=False) if self.use_history else None
        if self.use_history and self.use_context_adaptor:
            self.context_fusion = nn.Linear(2 * e, e)
            self.context_gate = nn.Linear(e, e)
        else:
            self.context_fusion = None
            self.context_gate = None
        self.decoder_initialization = nn.Linear(e + environmental_dim, self.num_layers * d)

        local_width = e if self.use_local_attention else 0
        if self.use_local_attention:
            self.observation_query = nn.Linear(d, e, bias=False)
            self.observation_key = nn.Linear(e, e, bias=False)
            self.observation_score = nn.Linear(e, 1, bias=False)
        else:
            self.observation_query = self.observation_key = self.observation_score = None
        decoder_input_dim = 2 + local_width + semantic_dim + int(self.query_time_conditioning)
        self.decoder = nn.GRU(decoder_input_dim, d, num_layers=self.num_layers,
                              batch_first=True, dropout=recurrent_dropout)
        self.decoder_input_dropout = nn.Dropout(probability)
        self.decoder_output_dropout = nn.Dropout(probability)
        self.coordinate_head = nn.Sequential(nn.Linear(d + local_width, head_dim),
                                             nn.ReLU(), nn.Linear(head_dim, 2))
        self.semantic_head = nn.Linear(d + local_width, 1)

    def _encode(self, coords: Tensor, times: Tensor, stays: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the same packed BiGRU and masked pooling to either source."""
        if coords.ndim != 3 or coords.shape[-1] != 2 or coords.shape[1] == 0:
            raise ValueError("Each encoded sequence needs at least one coordinate.")
        lengths = lengths.to(device=coords.device, dtype=torch.long)
        if lengths.shape != (coords.shape[0],) or (lengths < 1).any() or (lengths > coords.shape[1]).any():
            raise ValueError("Sequence lengths must be positive and within the padded width.")
        mask = torch.arange(coords.shape[1], device=coords.device)[None, :] < lengths[:, None]
        # Sanitize before linear layers: NaN padding would otherwise contaminate
        # parameter gradients even if its outputs were masked later.
        valid = mask.unsqueeze(-1)
        features = [coords.masked_fill(~valid, 0.0), times.masked_fill(~valid, 0.0)]
        if self.mobility_embedding is not None:
            features.append(self.mobility_embedding(stays.masked_fill(~valid, 0.0)))
        inputs = self.encoder_input_dropout(torch.cat(features, dim=-1))
        packed = pack_padded_sequence(inputs, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=coords.shape[1])
        states = self.encoder_output_dropout(states).masked_fill(~valid, 0.0)
        energies = torch.tanh(self.pool_projection(states)) @ self.pool_context
        weights = _masked_softmax(energies, mask)
        summary = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return states, summary, mask

    def _retrieve_history(self, batch: dict[str, Tensor], current: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b = current.shape[0]
        histories = batch.get("hist_coords")
        k = histories.shape[1] if histories is not None else 0
        weights = current.new_zeros((b, k))
        context = torch.zeros_like(current)
        available = torch.zeros(b, dtype=torch.bool, device=current.device)
        if not self.use_history or k == 0:
            return context, weights, available
        lengths = batch["hist_lengths"].to(device=current.device, dtype=torch.long)
        valid = batch["hist_mask"].to(device=current.device, dtype=torch.bool) & (lengths > 0)
        available = valid.any(dim=1)
        indices = valid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            return context, weights, available
        m = histories.shape[2]
        _, summaries, _ = self._encode(
            histories.reshape(b * k, m, 2).index_select(0, indices),
            batch["hist_time"].reshape(b * k, m, 1).index_select(0, indices),
            batch["hist_stay"].reshape(b * k, m, 1).index_select(0, indices),
            lengths.reshape(-1).index_select(0, indices))
        memory = current.new_zeros((b * k, self.encoder_width)).index_copy(0, indices, summaries)
        memory = memory.reshape(b, k, self.encoder_width)
        query = self.history_query(current)
        keys = self.history_key(memory)
        scores = (keys * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.encoder_width)
        weights = _masked_softmax(scores, valid)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights, available

    def forward(self, batch: dict[str, Tensor], teacher_forcing_ratio: float = 0.0) -> dict[str, Tensor]:
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("The teacher-forcing ratio must lie in [0, 1].")
        states, current, observed_mask = self._encode(
            batch["obs_coords"], batch["obs_time"], batch["obs_stay"], batch["obs_lengths"])
        history, history_weights, has_history = self._retrieve_history(batch, current)
        if self.context_fusion is not None:
            fused = torch.tanh(self.context_fusion(torch.cat((current, history), dim=-1)))
            joint = fused + fused * torch.sigmoid(self.context_gate(fused))
        else:
            joint = current + history
        joint = torch.where(has_history.unsqueeze(-1), joint, current)
        initial_features = [joint]
        if self.environment_embedding is not None:
            initial_features.append(self.environment_embedding(batch["environment"]))
        b = current.shape[0]
        hidden = torch.tanh(self.decoder_initialization(torch.cat(initial_features, dim=-1)))
        hidden = hidden.reshape(b, self.num_layers, self.decoder_hidden_size).transpose(0, 1).contiguous()
        target_mask = batch["target_mask"].to(device=current.device, dtype=torch.bool)
        target_time = batch["target_time"]
        if target_mask.ndim != 2 or target_mask.shape[0] != b or target_time.shape != (*target_mask.shape, 1):
            raise ValueError("Target times and validity mask must have shapes (B,T,1) and (B,T).")
        steps = target_mask.shape[1]
        if steps == 0:
            return {"coords": current.new_zeros((b, 0, 2)),
                    "stay_logits": current.new_zeros((b, 0)), "history_attention": history_weights}
        use_teacher_forcing = self.training and teacher_forcing_ratio > 0
        targets = None
        if use_teacher_forcing:
            targets = batch.get("target_coords")
            if targets is None or targets.shape != (b, steps, 2):
                raise ValueError("Coordinate teacher forcing requires target_coords with shape (B,T,2).")
        previous_coordinate = batch["obs_coords"][:, 0]
        previous_stay = batch["obs_stay"][:, 0]
        projected_keys = self.observation_key(states) if self.use_local_attention else None
        predictions, semantic_logits = [], []
        for index in range(steps):
            active = target_mask[:, index:index + 1]
            local = None
            decoder_features = [previous_coordinate]
            if self.use_local_attention:
                energies = self.observation_score(torch.tanh(
                    self.observation_query(hidden[-1]).unsqueeze(1) + projected_keys)).squeeze(-1)
                attention = _masked_softmax(energies, observed_mask)
                local = torch.bmm(attention.unsqueeze(1), states).squeeze(1)
                decoder_features.append(local)
            if self.mobility_embedding is not None:
                decoder_features.append(self.mobility_embedding(previous_stay))
            if self.query_time_conditioning:
                decoder_features.append(target_time[:, index].masked_fill(~active, 0.0))
            inputs = self.decoder_input_dropout(torch.cat(decoder_features, dim=-1))
            output, next_hidden = self.decoder(inputs.unsqueeze(1), hidden)
            hidden = torch.where(active.unsqueeze(0), next_hidden, hidden)
            head_features = [self.decoder_output_dropout(output.squeeze(1))]
            if local is not None:
                head_features.append(local)
            head_input = torch.cat(head_features, dim=-1)
            coordinate = self.coordinate_head(head_input)
            logit = self.semantic_head(head_input)
            predictions.append(coordinate.masked_fill(~active, 0.0))
            semantic_logits.append(logit.squeeze(-1).masked_fill(~active.squeeze(-1), 0.0))
            feedback = coordinate
            if use_teacher_forcing and index + 1 < steps:
                force = (torch.rand((b, 1), device=current.device) < teacher_forcing_ratio) & active
                feedback = torch.where(force, targets[:, index], coordinate)
            previous_coordinate = torch.where(active, feedback, previous_coordinate)
            # Semantic ground truth is supervision only, never a decoder input.
            previous_stay = torch.where(active, torch.sigmoid(logit), previous_stay)
        return {"coords": torch.stack(predictions, dim=1),
                "stay_logits": torch.stack(semantic_logits, dim=1),
                "history_attention": history_weights}
