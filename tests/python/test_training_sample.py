import pytest

from tamanobori.data import TrainingSample
from tamanobori.versions import MOVE_LABEL_SIZE


def test_training_sample_accepts_legal_move_labels() -> None:
    sample = TrainingSample(
        sample_id="sample-1",
        sfen="lnsgkgsnl/1r5b1/p1pppp1pp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1",
        legal_move_labels=(0, MOVE_LABEL_SIZE - 1),
    )

    sample.validate(MOVE_LABEL_SIZE)


def test_training_sample_rejects_empty_legal_moves() -> None:
    sample = TrainingSample(sample_id="sample-1", sfen="sfen", legal_move_labels=())

    with pytest.raises(ValueError, match="must not be empty"):
        sample.validate(MOVE_LABEL_SIZE)


def test_training_sample_rejects_out_of_range_move_labels() -> None:
    sample = TrainingSample(
        sample_id="sample-1",
        sfen="sfen",
        legal_move_labels=(MOVE_LABEL_SIZE,),
    )

    with pytest.raises(ValueError, match="out of range"):
        sample.validate(MOVE_LABEL_SIZE)
