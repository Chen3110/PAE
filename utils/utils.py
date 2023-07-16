import torch
from torch import nn

def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros([total_size], device=rvecs.device))  # for broadcasting
    Ks_1 = torch.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = torch.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = torch.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(torch.eye(3, device=rvecs.device).repeat(total_size,1,1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
         (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:,None,None], identity_mat, Rs)

    return R.reshape(batch_size, -1)


def rotation6d_2_rot_mat(rotation6d):
    batch_size = rotation6d.shape[0]
    pose6d = rotation6d.reshape(-1, 6)
    tmp_x = nn.functional.normalize(pose6d[:,:3], dim = -1)
    tmp_z = nn.functional.normalize(torch.cross(tmp_x, pose6d[:,3:], dim = -1), dim = -1)
    tmp_y = torch.cross(tmp_z, tmp_x, dim = -1)

    tmp_x = tmp_x.view(-1, 3, 1)
    tmp_y = tmp_y.view(-1, 3, 1)
    tmp_z = tmp_z.view(-1, 3, 1)
    R = torch.cat((tmp_x, tmp_y, tmp_z), -1)

    return R.reshape(batch_size, -1)
