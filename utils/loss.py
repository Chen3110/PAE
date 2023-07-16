import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

from torch import nn

class LossManager():
    def __init__(self, ding_bot=None) -> None:
        super(LossManager).__init__()
        self.loss_dict = {}
        self.batch_loss = []
        self.ding_bot = ding_bot

    def update_loss(self, name, loss):
        if name not in self.loss_dict:
            self.loss_dict.update({name:[loss]})
        else:
            self.loss_dict[name].append(loss)

    def calculate_total_loss(self):
        batch_loss = []
        for loss in self.loss_dict.values():
            batch_loss.append(loss[-1])
        if isinstance(loss[-1], torch.Tensor):
            total_loss = torch.sum(torch.stack(batch_loss))
        else:
            total_loss = np.sum(np.stack(batch_loss))
        self.batch_loss.append(total_loss)
        return total_loss

    def calculate_epoch_loss(self, output_path, epoch):
        fig = plt.figure()
        loss_json = os.path.join(output_path, "loss.json")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            open(loss_json, "w")
        with open(loss_json, "r") as f:
            losses = json.load(f) if epoch - 1 else dict()
        with open(loss_json, "w") as f:
            losses.update({"epoch":epoch})
            for i, (name, loss) in enumerate(self.loss_dict.items()):
                epoch_loss = np.average(torch.tensor(loss))
                loss_list = np.hstack((losses.get(name, [])[:epoch-1], epoch_loss))
                losses.update({name:list(loss_list)})
                fig.add_subplot(3, 3, i+1, title=name).plot(loss_list)
            total_loss = np.hstack((losses.get("total_loss", [])[:epoch-1], np.average(torch.tensor(self.batch_loss))))
            losses.update({"total_loss":list(total_loss)})
            json.dump(losses, f)
        fig.add_subplot(3, 3, i+2, title="total_loss").plot(total_loss)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close()
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        if self.ding_bot is not None:
            self.ding_bot.add_md("train_immfusion", "【IMG】 \n ![img]({}) \n 【{}】\n epoch={}, loss={}".format(self.ding_bot.img2b64(img), output_path, epoch, total_loss[-1]))
            self.ding_bot.enable()
        
        self.loss_dict = {}
        self.batch_loss = []

    def calculate_test_loss(self, output_path):
        fig = plt.figure()
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        loss_json = os.path.join(output_path, "loss.json")
        losses = {}
        for i, (name, loss) in enumerate(self.loss_dict.items()):
            _loss = np.sort(torch.tensor(loss))
            losses.update({name:np.mean(_loss).tolist()})
            hist, bin_edges = np.histogram(_loss, bins=100)
            cdf = np.cumsum(hist)/len(_loss)
            fig.add_subplot(2, 3, i+1, title=name).plot(bin_edges[:-1], cdf)
        total_loss = np.average(torch.tensor(self.batch_loss))
        losses.update({"total_loss":total_loss.tolist()})
        with open(loss_json, "w") as f:
            json.dump(losses, f)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close()
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        np.save(os.path.join(output_path, "joints_loss"), np.sort(torch.tensor(self.loss_dict["joints_loss"])))
        np.save(os.path.join(output_path, "vertices_loss"), np.sort(torch.tensor(self.loss_dict["vertices_loss"])))


class GeodesicLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(GeodesicLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            # breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta


def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    if len(gt_keypoints_3d) > 0:
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 