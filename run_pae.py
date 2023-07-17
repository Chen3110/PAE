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
from utils.utils import *
import visualization.plotting as plot
from utils.loss import GeodesicLoss, LossManager, keypoint_3d_loss

import numpy as np
import torch
import random

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
        label = utility.ReadBatchFromMatrix(Label, gather.flatten())

        batch = batch.reshape(shape[0], shape[1], -1)
        batch = batch.permute(0, 2, 1)
        batch = batch.reshape(shape[0], batch.shape[1]*batch.shape[2])
        label = label.reshape(shape[0], shape[1], -1)
        label = label.permute(0, 2, 1)
        label = label.reshape(shape[0], label.shape[1]*label.shape[2])
        return batch, label
    
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
        with open('data/train_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        Frames = data_dict['frames']
        Data = data_dict[args.input_type].reshape(Frames.shape[0], -1).astype(np.float32)
        if args.input_type == 'joints_3d':
            Data -= Data[:,0:1]
        Betas = data_dict['betas']
        Label = data_dict['pose'].copy()
    else:
        # transform joints_3d to projected coordinats
        trans_mat = dict(
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            t = [0, -3, 0]
        )
        data_dict = dict(
            pose = [],
            shape_data = [],
            sequences = [],
            joints_3d = [],
            joints_2d = [],
        )
        num_frames_all = []
        
        for i, seq_file in enumerate(all_seq_files):
            data = dict(np.load(seq_file, allow_pickle=True))
            num_frames = data['pose_body'].shape[0]
            pose = np.hstack((data['root_orient'], data['pose_body']))
            data_dict['pose'].append(pose.astype(np.float32))
            joints_3d = data['joints'] - data['joints'][:,0:1]
            data_dict['joints_3d'].append(joints_3d.reshape(num_frames,-1).astype(np.float32))
            joints_3d_trans = (joints_3d - trans_mat['t']) @ trans_mat['R']
            joints_2d = project_pcl_torch(joints_3d_trans, image_size=[1000,1002]).numpy()
            data_dict['joints_2d'].append(joints_2d.reshape(num_frames,-1).astype(np.float32))
            data_dict['shape_data'].append(data['betas'])
            data_dict['sequences'] += [i+1] * num_frames
            num_frames_all.append(num_frames)
        Label = np.vstack(data_dict['pose'])
        Frames = np.array(data_dict['sequences'])
        Betas = np.array(data_dict['shape_data'])
        Data = np.vstack(data_dict[args.input_type])
        
    if args.input_dim == 9:
        Data = rodrigues_2_rot_mat(torch.from_numpy(Data).float()).numpy()
    Save = args.output_path
    utility.MakeDirectory(Save + '/checkpoints')
    utility.MakeDirectory(Save + '/parameters')
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

    dist_amps = []
    dist_freqs = []
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
    if args.resume_checkpoint!=None:
        network = torch.load(os.path.join(args.resume_checkpoint))
        resume_epoch = int(os.path.basename(args.resume_checkpoint).split('_')[0])
    else:
        network = model.Model(
            input_channels=input_channels,
            embedding_channels=phase_channels,
            time_range=frames,
            window=window,
            output_channels=output_channels,
        )
        resume_epoch = 0
    network.to(device)
    # evaluate
    if not args.train:
        print("Eval Phases")
        # if only run eval, load checkpoint
        vis = EvaluateStreamPlot()
        per_joint_err = []
        per_vertex_err = []
        per_pampjpe = []
        for i in range(0, sample_count, batch_size):
            # utility.PrintProgress(i, sample_count, sample_count/batch_size)
            eval_indices = I[i:i+batch_size]
            betas = Betas[Frames[eval_indices]-1]
            #Run model prediction
            network.eval()
            body_model.eval()
            eval_batch, eval_label = LoadBatches(eval_indices)
            yPred, _, _, _ = network(eval_batch)

            batches = eval_batch.shape[0]
            yPred = yPred.view(batches, -1, frames).permute(0, 2, 1).contiguous()
            if args.output_dim == 6:
                yPred = rotation6d_2_rot_mat(yPred).view(batches, frames, -1)
            eval_label = eval_label.view(batches, -1, frames).permute(0, 2, 1).contiguous()
            eval_label = rodrigues_2_rot_mat(eval_label).view(batches, frames, -1)
            # calculate errors
            beta = torch.tensor(betas, device=device).float()
            pred_mesh = body_model(yPred[:,-1], beta)
            label_mesh = body_model(eval_label[:,-1], beta)
            pred_vertices = pred_mesh['verts']
            gt_vertices = label_mesh['verts']
            pred_3d_joints = pred_mesh['joints']
            gt_3d_joints = label_mesh['joints']
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
            error_joints = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints)
            error_joints_pa = reconstruction_error(copy2cpu(pred_3d_joints), copy2cpu(gt_3d_joints[:,:,:3]), reduction=None)
            per_joint_err.append(error_vertices)
            per_vertex_err.append(error_joints)
            per_pampjpe.append(error_joints_pa)
            if args.visual:
                for b in range(batches):
                    gen = mesh_generator(copy2cpu(pred_vertices[b]), copy2cpu(gt_vertices[b]), copy2cpu(pred_mesh['faces']))
                    vis.show(gen, fps=30)
        # save errors
        output_path = os.path.join(args.output_path, 'error')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        j_err = np.hstack(per_joint_err)
        v_err = np.hstack(per_vertex_err)
        pa_err = np.hstack(per_pampjpe)
        np.save(os.path.join(output_path, "per_joint_err"), j_err)
        np.save(os.path.join(output_path, "per_vertex_err"), v_err)
        np.save(os.path.join(output_path, "pampjpe"), pa_err)
        print("mean joint err (cm):", np.mean(j_err)*100)
        print("mean vertex err (cm):", np.mean(v_err)*100)
        print("pampjpe (cm):", np.mean(pa_err)*100)
        with open(os.path.join(output_path, "error.txt"), 'w') as f:
            f.write("mean joint error: " + str(np.mean(j_err)*100))
            f.write("\nmean vertex error: " + str(np.mean(v_err)*100))
            f.write("\npampjpe: " + str(np.mean(pa_err)*100))
    # train the model
    else:
        #Build network model
        print("Training Phases")
        #Setup optimizer and loss function
        optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
        criterion_3d_joints = torch.nn.MSELoss(reduction='none').to(device)
        criterion_vertices = torch.nn.L1Loss().to(device)
        
        for epoch in range(epochs):
            epoch += 1 + resume_epoch
            scheduler.step()
            rng.shuffle(I)
            start_time = time.time()
            for i in range(0, sample_count, batch_size):
                # utility.PrintProgress(i, sample_count, sample_count/batch_size)
                train_indices = I[i:i+batch_size]
                betas = torch.tensor(Betas[Frames[train_indices]-1]).float().to(device)

                #Run model prediction
                network.train()
                train_batch, train_label = LoadBatches(train_indices)
                yPred, latent, signal, params = network(train_batch)
                batches = train_batch.shape[0]
                yPred = yPred.view(batches, -1, frames).permute(0, 2, 1).contiguous()
                if args.output_dim == 6:
                    yPred = rotation6d_2_rot_mat(yPred).view(batches, frames, -1)
                pred_mesh = body_model(yPred[:,-1], betas)
                train_label = train_label.view(batches, -1, frames).permute(0, 2, 1).contiguous()
                train_label = rodrigues_2_rot_mat(train_label).view(batches, frames, -1)
                gt_mesh = body_model(train_label[:,-1], betas)
                    
                #Compute loss and train
                pose_loss = loss_function(yPred, train_label)
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
                _a_ = copy2cpu(params[2]).squeeze()
                for i in range(_a_.shape[0]):
                    dist_amps.append(_a_[i,:])
                while len(dist_amps) > 10000:
                    dist_amps.pop(0)

                _f_ = copy2cpu(params[1]).squeeze()
                for i in range(_f_.shape[0]):
                    dist_freqs.append(_f_[i,:])
                while len(dist_freqs) > 10000:
                    dist_freqs.pop(0)

            torch.save(network, Save + "/checkpoints/"+str(epoch)+"_"+str(phase_channels)+"Channels"+".pt") 

            print('Epoch', epoch, 'loss', copy2cpu(loss))
            loss_manager.calculate_epoch_loss(os.path.join(args.output_path, 'loss/train'), epoch)

            #Save Phase Parameters
            print("Saving Parameters")
            network.eval()
            E = np.arange(sample_count)
            with open(Save+'/parameters/Parameters_'+str(epoch)+'.txt', 'w') as file:
                for i in range(0, sample_count, batch_size):
                    # utility.PrintProgress(i, sample_count)
                    eval_indices = E[i:i+batch_size]
                    eval_batch, _ = LoadBatches(eval_indices)
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
    parser.add_argument('--amass_root', type=str, default='./data/amass_processed', help='Root directory of AMASS dataset.')
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