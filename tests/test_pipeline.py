"""Testes unitários do pipeline (sem rede nem BLAST)."""

import numpy as np
import pytest

from cb396.dataset import dssp_to_q3_char
from cb396.features import make_sliding_windows


def test_dssp_to_q3():
    assert [dssp_to_q3_char(c) for c in "HGIEBCST-"] == list("HHHEECCCC")


def test_sliding_window_shape_and_center():
    # PSSM 5 resíduos x 20 scores; cada linha = índice do resíduo
    pssm = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 20))
    q3 = "HHEEC"
    X, y = make_sliding_windows(pssm, q3, window_size=13)

    assert X.shape == (5, 13 * 20)
    assert list(y) == [0, 0, 1, 1, 2]  # H,H,E,E,C

    # a posição central (índice 6 da janela) deve ser o próprio resíduo
    center = X.reshape(5, 13, 20)[:, 6, 0]
    assert list(center) == [0, 1, 2, 3, 4]

    # bordas com padding zero (primeira janela: 6 posições iniciais são zero)
    assert X.reshape(5, 13, 20)[0, 0, 0] == 0.0


def test_sliding_window_rejects_even():
    pssm = np.zeros((3, 20), dtype=np.float32)
    try:
        make_sliding_windows(pssm, "CCC", window_size=12)
    except ValueError:
        return
    raise AssertionError("window_size par deveria falhar")


def test_cnn_reconstruction_recovers_center():
    """A coluna central da janela deve reconstruir a PSSM original do resíduo."""
    pytest.importorskip("torch")
    from cb396.cnn import load_protein_tensors_from_features
    from cb396.config import Paths

    # PSSM 4 resíduos: linha r = valor r; rótulos C
    pssm = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 20))
    X, y = make_sliding_windows(pssm, "CCCC", window_size=13)

    # injeta features sem tocar no disco (monkeypatch de load_features)
    import cb396.cnn as cnn

    orig = cnn.load_features
    cnn.load_features = lambda _paths: (X, y, np.array(["p1"] * 4))
    try:
        items = load_protein_tensors_from_features(Paths(), normalize=False)
    finally:
        cnn.load_features = orig

    assert len(items) == 1
    _id, x, yy = items[0]
    assert x.shape == (4, 20)
    assert list(x[:, 0]) == [0, 1, 2, 3]  # centro = PSSM original
    assert list(yy) == [2, 2, 2, 2]


def test_crf_decode_and_nll():
    """Com transições zeradas, o Viterbi do CRF reduz a argmax por posição."""
    pytest.importorskip("torch")
    import torch

    from cb396.cnn import CRF

    crf = CRF(3)
    with torch.no_grad():
        crf.transitions.zero_()
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()

    emissions = torch.tensor(
        [[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0], [5.0, 0, 0]]]
    )  # (1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)

    assert crf.decode(emissions, mask)[0] == [0, 1, 2, 0]

    tags = torch.tensor([[0, 1, 2, 0]])
    loss = crf.nll(emissions, tags, mask)
    assert torch.isfinite(loss) and loss.item() > 0
