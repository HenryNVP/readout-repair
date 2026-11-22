import pytest

from readout_repair.act import low_rank


def test_low_rank_starts_at_parent_and_has_rank_r():
    torch = pytest.importorskip('torch')
    lr = low_rank((6, 5), r=2, seed=0)
    W = torch.randn(6, 5)
    assert torch.equal(lr(W), W)
    with torch.no_grad():
        lr.B.normal_()
    assert torch.linalg.matrix_rank(lr(W) - W) == 2
