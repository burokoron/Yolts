from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    sfen: str
    legal_move_labels: tuple[int, ...]

    def validate(self, move_label_size: int) -> None:
        if not self.legal_move_labels:
            msg = "legal_move_labels must not be empty"
            raise ValueError(msg)

        out_of_range = [
            label
            for label in self.legal_move_labels
            if label < 0 or label >= move_label_size
        ]
        if out_of_range:
            msg = f"move labels out of range: {out_of_range}"
            raise ValueError(msg)
