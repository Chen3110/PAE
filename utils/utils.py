import torch
from torch import nn
import numpy as np

INTRINSIC = {
    'image4': np.asarray([
        [1145.51133842318,      0,                  514.968197319863],
        [0,                     1144.77392807652,   501.882018537695],
        [0,                     0,                  1]]),
}

def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()

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


def project_pcl(pcl, trans_mat=None, intrinsic=None, image_size=[1536,2048]):
    if trans_mat is not None:
        pcl = (pcl - trans_mat['t']) @ np.array(trans_mat['R']).reshape(3, 3)
    intrinsic = np.array(intrinsic) if intrinsic is not None else INTRINSIC['image4']
    pcl_2d = ((pcl/pcl[:,2:3]) @ intrinsic.T)[:,:2]
    pcl_2d = np.floor(pcl_2d).astype(int)
    pcl_2d[:, [0, 1]] = pcl_2d[:, [1, 0]]
    # filter out the points exceeding the image size
    image_size = np.array(image_size)
    pcl_2d = np.where(pcl_2d<image_size-1, pcl_2d, image_size-1)
    pcl_2d = np.where(pcl_2d>0, pcl_2d, 0)
    
    return pcl_2d

def project_pcl_torch(pcl, trans_mat=None, intrinsic=None, image_size=[1536,2048]):
    """
    Project pcl to the image plane
    """
    if trans_mat is not None:
        pcl = (pcl - trans_mat[:,None,:3,3]) @ trans_mat[:,:3,:3]
    intrinsic = intrinsic if intrinsic is not None else INTRINSIC['image4']
    if type(pcl) != torch.Tensor:
        pcl = torch.tensor(pcl).float()
    pcl_2d = ((pcl/pcl[:,:,2:3]) @ torch.tensor(intrinsic).T.float().to(pcl.device))[:,:,:2]
    pcl_2d = torch.floor(pcl_2d).long()
    pcl_2d[:,:,[0,1]] = pcl_2d[:,:,[1,0]]
    image_size = torch.tensor(image_size).to(pcl.device)
    pcl_2d = torch.where(pcl_2d<image_size-1, pcl_2d, image_size-1)
    pcl_2d = torch.where(pcl_2d>0, pcl_2d, 0)
    return pcl_2d

def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def mean_per_joint_position_error(pred, gt):
    """ 
    Compute mPJPE
    """
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt):
    """
    Compute mPVE
    """
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt):
    """
    Compute mPVE
    """
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    # conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    if len(gt_keypoints_3d) > 0:
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    if len(gt_vertices) > 0:
        return criterion_vertices(pred_vertices, gt_vertices)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 