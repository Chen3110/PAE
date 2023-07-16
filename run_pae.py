import argparse
import glob
import os
import pickle
import time
import visualization.library.Utility as utility
import visualization.library.AdamWR.adamw as adamw
import visualization.library.AdamWR.cyclic_scheduler as cyclic_scheduler
import network.pae as model
from visualization.visual import EvaluateStreamPlot
from utils.smpl_model import SMPLXModel
from utils.utils import rodrigues_2_rot_mat, rotation6d_2_rot_mat
import visualization.plotting as plot
from utils.loss import GeodesicLoss, LossManager, keypoint_3d_loss

import numpy as np
import torch
import random

import matplotlib.pyplot as plt

TRAIN_DATASETS = ['ACCAD', 'BMLmovi', 'BioMotionLab_NTroje', 'BMLhandball', 'CMU', 'DanceDB', 'DFaust_67', 
                'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'MPI_HDM05', 
                'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap', 
                'TotalCapture', 'Transitions_mocap']
TEST_DATASETS = ['HUMAN4D']

def main(args):
    torch.cuda.set_device(args.gpu_idx)
    device = torch.device('cuda')
    #Start Parameter Section
    window = 2.0 #time duration of the time window
    fps = 30 #fps of the motion capture data
    joints = 22 #joints of the character skeleton

    frames = int(window * fps) + 1
    input_channels = args.input_dim*joints #number of channels along time in the input data (here 3*J as XYZ-component of each joint)
    output_channels = args.output_dim*joints #number of channels along time in the input data (here 3*J as XYZ-component of each joint)
    phase_channels = 5 #desired number of latent phase channels (usually between 2-10)

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    plotting_interval = 500 #update visualization at every n-th batch (visualization only)
    pca_sequence_count = 100 #number of motion sequences visualized in the PCA (visualization only)
    test_sequence_ratio = 0.01 #ratio of randomly selected test sequences (visualization only)
    #End Parameter Section

    def LoadBatches(indices):
        gather = gather_window.reshape(1,-1).repeat(indices.shape[0], 0)

        data = data_frames[indices]

        pivot = data[:,0].reshape(-1,1)
        min = data[:,1].reshape(-1,1)
        max = data[:,2].reshape(-1,1)

        gather = np.clip(gather + pivot, min, max)

        shape = gather.shape

        batch = utility.ReadBatchFromMatrix(Data, gather.flatten())

        batch = batch.reshape(shape[0], shape[1], -1)
        batch = batch.permute(0, 2, 1)
        batch = batch.reshape(shape[0], batch.shape[1]*batch.shape[2])
        return batch

    def Item(value):
        return value.detach().cpu()
    
    # requested datasets
    if args.datasets is None:
        args.datasets = TRAIN_DATASETS if args.train else TEST_DATASETS
    dataset_dirs = [os.path.join(args.amass_root, name) for name in args.datasets]
    print('Requested datasets:', args.datasets)
    
    all_seq_files = []
    for data_dir in dataset_dirs:
        input_seqs = glob.glob(os.path.join(data_dir, '*/*30_fps.npz'))
        all_seq_files += input_seqs
        
    print('Sequences:', len(all_seq_files))
    
    if args.load_data:
        with open('data/data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        Data = data_dict[args.input_type]
        Frames = data_dict['frames']
        Betas = data_dict['betas']
    else:
        pose_data = []
        shape_data = []
        sequences = []
        num_frames_all = []
        for i, seq_file in enumerate(all_seq_files):
            data = dict(np.load(seq_file, allow_pickle=True))
            num_frames = data['pose_body'].shape[0]
            pose = np.hstack((data['root_orient'], data['pose_body']))
            if args.input_dim == 3:
                pose_data.append(pose)
            else:
                rot_mat = rodrigues_2_rot_mat(torch.from_numpy(pose).float())
                pose_data.append(rot_mat.numpy())
            shape_data.append(data['betas'])
            sequences += [i+1] * num_frames
            num_frames_all.append(num_frames)
        Data = np.vstack(pose_data).astype(np.float32)
        Frames = np.array(sequences)
        Betas = np.array(shape_data)
    
    Save = args.output_path
    utility.MakeDirectory(Save)
    gather_padding = (int((frames-1)/2))
    gather_window = np.arange(frames) - gather_padding

    #Start Generate Data Frames
    print("Generating Data Frames")
    data_frames = []
    test_frames = []

    for i in range(Frames[-1]):
        # utility.PrintProgress(i, Frames[-1])
        indices = np.where(Frames == (i+1))[0]
        for j in range(indices.shape[0]):
            slice = [indices[j], indices[0], indices[-1]]
            data_frames.append(slice)
            if np.random.uniform(0, 1) < test_sequence_ratio and indices[j] >= (indices[0]+gather_padding) and indices[j] <= (indices[-1]-gather_padding):
                test_frames.append(j)

    print("Data Frames:", len(data_frames))
    if args.train:
        print("Test Frames:", len(test_frames))
    data_frames = np.array(data_frames)
    sample_count = len(data_frames)
    #End Generate Data Frames

    #Initialize all seeds
    seed = 23456
    rng = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    #Initialize drawing
    # plt.ion()
    # _, ax1 = plt.subplots(6,1)
    # _, ax2 = plt.subplots(phase_channels,5)
    # _, ax3 = plt.subplots(1,2)
    # _, ax4 = plt.subplots(2,1)
    dist_amps = []
    dist_freqs = []
    # loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)
    loss_manager = LossManager()

    def mesh_generator(pred_verts, label_verts, faces):
        result = dict(
            pred_smpl = dict(
                mesh = [pred_verts, faces],
                color = np.asarray([179, 230, 213]) / 255
            ),
            label_smpl = dict(
                mesh = [label_verts, faces],
                color = np.asarray([235, 189, 191]) / 255
            )
        )
        yield result
        
    I = np.arange(sample_count)
    loss_function = torch.nn.MSELoss()
    # loss_function = GeodesicLoss()
    
    body_model = SMPLXModel(bm_fname='/home/nesc525/drivers/0/chen/3DSVC/ignoredata/mosh_files/smplx/neutral/model.npz', 
                            num_betas=16, num_expressions=0, device=device)
    # evaluate
    if not args.train and args.resume_checkpoint!=None:
        print("Eval Phases")
        # if only run eval, load checkpoint
        network = torch.load(os.path.join(args.resume_checkpoint))
        vis = EvaluateStreamPlot()
        for i in range(0, sample_count, batch_size):
            # utility.PrintProgress(i, sample_count, sample_count/batch_size)
            eval_indices = I[i:i+batch_size]
            betas = Betas[Frames[eval_indices]-1]
            #Run model prediction
            network.eval()
            eval_batch = LoadBatches(eval_indices)
            yPred, _, _, _ = network(eval_batch)
            if args.visual:
                yPred = yPred.view(eval_batch.shape[0], -1, frames).permute(0, 2, 1).contiguous()
                eval_batch = eval_batch.view(eval_batch.shape[0], -1, frames).permute(0, 2, 1).contiguous()
                if args.output_dim == 6:
                    yPred = rotation6d_2_rot_mat(yPred).view(eval_batch.shape[0], frames, -1)
                if args.input_dim == 3:
                    eval_batch = rodrigues_2_rot_mat(eval_batch).view(eval_batch.shape[0], frames, -1)
                for b in range(batch_size):
                    beta = torch.tensor(betas[b:b+1], device=device).float()
                    pred_mesh = body_model(yPred[b][0:1], beta)
                    label_mesh = body_model(eval_batch[b][0:1], beta)
                    gen = mesh_generator(Item(pred_mesh['verts'][0]), Item(label_mesh['verts'][0]), Item(pred_mesh['faces']))
                    vis.show(gen, fps=30)
    else:
        #Build network model
        print("Training Phases")
        network = utility.ToDevice(model.Model(
            input_channels=input_channels,
            embedding_channels=phase_channels,
            time_range=frames,
            window=window,
            output_channels=output_channels
        ))
        #Setup optimizer and loss function
        optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
        criterion_3d_joints = torch.nn.MSELoss(reduction='none').to(device)
        criterion_vertices = torch.nn.L1Loss().to(device)
        
        for epoch in range(epochs):
            scheduler.step()
            rng.shuffle(I)
            start_time = time.time()
            for i in range(0, sample_count, batch_size):
                utility.PrintProgress(i, sample_count, sample_count/batch_size)
                train_indices = I[i:i+batch_size]
                betas = torch.tensor(Betas[Frames[train_indices]-1]).float().to(device)
                # betas = betas[:,None,:].repeat(1,frames,1).reshape(-1,16)

                #Run model prediction
                network.train()
                train_batch = LoadBatches(train_indices)
                yPred, latent, signal, params = network(train_batch)
                yPred = yPred.view(train_batch.shape[0], -1, frames).permute(0, 2, 1).contiguous()
                train_batch = train_batch.view(train_batch.shape[0], -1, frames).permute(0, 2, 1).contiguous()
                if args.output_dim == 6:
                    yPred = rotation6d_2_rot_mat(yPred).view(train_batch.shape[0], frames, -1)
                pred_mesh = body_model(yPred[:,0], betas)
                if args.input_dim == 3:
                    train_batch = rodrigues_2_rot_mat(train_batch).view(train_batch.shape[0], frames, -1)
                gt_mesh = body_model(train_batch[:,0], betas)
                    
                #Compute loss and train
                pose_loss = loss_function(yPred, train_batch)
                loss_3d_joints = args.loss_weight * keypoint_3d_loss(criterion_3d_joints, pred_mesh['joints'], gt_mesh['joints'], device)
                loss_vertices = args.loss_weight * keypoint_3d_loss(criterion_vertices, pred_mesh['verts'], gt_mesh['verts'], device)
                
                loss_manager.update_loss("Pose Loss", pose_loss)
                loss_manager.update_loss("Joints Loss", loss_3d_joints)
                loss_manager.update_loss("Vertices Loss", loss_vertices)
                loss = loss_manager.calculate_total_loss()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.batch_step()

                #Start Visualization Section
                _a_ = Item(params[2]).squeeze().numpy()
                for i in range(_a_.shape[0]):
                    dist_amps.append(_a_[i,:])
                while len(dist_amps) > 10000:
                    dist_amps.pop(0)

                _f_ = Item(params[1]).squeeze().numpy()
                for i in range(_f_.shape[0]):
                    dist_freqs.append(_f_[i,:])
                while len(dist_freqs) > 10000:
                    dist_freqs.pop(0)

            torch.save(network, Save + "/"+str(epoch+1)+"_"+str(phase_channels)+"Channels"+".pt") 

            print('Epoch', epoch+1, 'loss', Item(loss).numpy())
            loss_manager.calculate_epoch_loss(os.path.join(args.output_path, 'loss/train'), epoch+1)

            #Save Phase Parameters
            print("Saving Parameters")
            network.eval()
            E = np.arange(sample_count)
            with open(Save+'/Parameters_'+str(epoch+1)+'.txt', 'w') as file:
                for i in range(0, sample_count, batch_size):
                    # utility.PrintProgress(i, sample_count)
                    eval_indices = E[i:i+batch_size]
                    eval_batch = LoadBatches(eval_indices)
                    _, _, _, params = network(eval_batch)
                    p = utility.ToNumpy(params[0]).squeeze()
                    f = utility.ToNumpy(params[1]).squeeze()
                    a = utility.ToNumpy(params[2]).squeeze()
                    b = utility.ToNumpy(params[3]).squeeze()
                    for j in range(p.shape[0]):
                        params = np.concatenate((p[j,:],f[j,:],a[j,:],b[j,]))
                        line = ' '.join(map(str, params))
                        if (i+j) == (sample_count-1):
                            file.write(line)
                        else:
                            file.write(line + '\n')
            end_time = time.time() - start_time
            print("Epoch Time:", time.strftime("%H h %M m %S s", time.gmtime(end_time)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amass_root', type=str, default='/home/nesc525/drivers/4/chen/humor/data/amass_processed', help='Root directory of AMASS dataset.')
    parser.add_argument('--datasets', type=str, nargs='+', default=None, help='Which datasets to process. By default processes all.')
    parser.add_argument('--output_path', type=str, default='./output', help='Root directory to save output to.')
    parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size.")
    parser.add_argument("--gpu_idx", type=int, default=0, help="select gpu")
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')
    parser.add_argument("--resume_checkpoint", default=None, type=str, help="Path to specific checkpoint for resume training.")
    parser.add_argument('--visual', dest="visual", action="store_true", help='visualize mesh')
    parser.add_argument("--loss_weight", default=1, type=int, help="loss weight")
    parser.add_argument("--input_dim", default=3, type=int, help="input dimension")
    parser.add_argument("--output_dim", default=6, type=int, help="output dimension")
    parser.add_argument('--load_data', action="store_true", help='load data')
    parser.add_argument('--input_type', default='pose', type=str, help="Input type: pose, joints_3d or joints_2d")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)