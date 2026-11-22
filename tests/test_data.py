import numpy as np

from readout_repair.data import chunks, replay_frames


def test_replay_frames_pads_chunks_with_last_action():
    ep = dict(img=[np.zeros((3, 2, 2, 3), np.uint8)] * 2, state=np.zeros((3, 9), np.float32),
              act=np.arange(3, dtype=np.float32)[:, None] * np.ones((1, 10)))
    img, state, chunk, pad, epi = replay_frames([ep], H=4)
    assert chunk.shape == (3, 4, 10) and len(img[0]) == 3
    assert np.allclose(chunk[1, :, 0], [1, 2, 2, 2]) and pad[1].tolist() == [0, 0, 1, 1]


def test_chunks_repeat_the_last_action():
    c = chunks(np.arange(3.0)[:, None], 4)
    assert c.shape == (3, 4, 1) and c[1, :, 0].tolist() == [1, 2, 2, 2]
