"""Stand-in for the four pytorch3d.transforms functions DP's RotationTransformer uses (axis_angle
<-> rotation_6d); pytorch3d has no wheel for this torch. Same conventions as pytorch3d: 6D = the
first two ROWS of R, Gram-Schmidt back. Checked against evh_controller/rotation.py in tests/test_dp_shim.py."""
import torch
from scipy.spatial.transform import Rotation


def axis_angle_to_matrix(a):
    m = Rotation.from_rotvec(a.reshape(-1, 3).detach().cpu().double().numpy()).as_matrix()
    return torch.as_tensor(m, dtype=a.dtype, device=a.device).reshape(*a.shape[:-1], 3, 3)


def matrix_to_axis_angle(m):
    a = Rotation.from_matrix(m.reshape(-1, 3, 3).detach().cpu().double().numpy()).as_rotvec()
    return torch.as_tensor(a, dtype=m.dtype, device=m.device).reshape(*m.shape[:-2], 3)


def matrix_to_rotation_6d(m):
    return m[..., :2, :].clone().reshape(*m.shape[:-2], 6)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)
