import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, read_pickle, save_new_pickle, generateZ, calculate_iou
from utils import calculate_chamfer_distance
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import ShapeNetMultiviewDataset, var_or_cuda
from model import _G, _D, _E_MultiView
plt.switch_backend("TkAgg")

def KLLoss(z_mu,z_var):
    return (- 0.5 * torch.sum(1 + z_var - torch.pow(z_mu, 2) - torch.exp(z_var)))

def train_multiview(args):
    hyparam_list = [("model", args.model_name), ("attention", args.attention_type)]
    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    if args.use_tensorboard:
        import tensorflow as tf
        summary_writer = tf.summary.create_file_writer(args.output_dir + args.log_dir + log_param)
        def inject_summary(summary_writer, tag, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetMultiviewDataset(dsets_path, args, is_train=True)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last=True)

    D = _D(args)
    G = _G(args)
    E = _E_MultiView(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)
    
    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()
        E.cuda()

    criterion = nn.BCELoss()

    pickle_path = args.output_dir + args.pickle_dir + log_param
    start_epoch = read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver) + 1
    print(f"Eğitime {start_epoch}. epoktan devam ediliyor.")

    for epoch in range(start_epoch, args.n_epochs+1):
        epoch_start_time = time.time()
        epoch_iou, epoch_chamfer_distance = 0, 0
        epoch_d_precision, epoch_d_recall, epoch_d_f1 = 0, 0, 0
        epoch_d_real_loss, epoch_d_fake_loss, epoch_d_loss, epoch_g_loss = 0, 0, 0, 0
        epoch_d_acu, epoch_recon_loss, epoch_kl_loss = 0, 0, 0
        num_batches, num_cd_valid_batches = 0, 0

        for i, (images, model_3d) in enumerate(dset_loaders):
            images = [var_or_cuda(img) for img in images]
            model_3d = var_or_cuda(model_3d)

            Z = generateZ(args)
            Z_vae, z_mus, z_vars = E(images)
            G_vae = G(Z_vae)
            
            real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))
            fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

            d_real = D(model_3d)
            d_real_loss = criterion(d_real.squeeze(), real_labels)
            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake.squeeze(), fake_labels)
            d_loss = d_real_loss + d_fake_loss
            # print(f"dreal= {d_real_loss* 15000}")
            # print(f"dfake= {d_fake_loss* 15000}")

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            y_true_d = torch.cat((torch.ones_like(real_labels), torch.zeros_like(fake_labels)), 0)
            y_pred_d_prob = torch.cat((d_real.squeeze(), d_fake.squeeze()), 0)
            y_pred_d = (y_pred_d_prob >= 0.5).float()
            y_true_d_np = y_true_d.cpu().numpy()
            y_pred_d_np = y_pred_d.cpu().numpy()

            precision = precision_score(y_true_d_np, y_pred_d_np, zero_division=0)
            recall = recall_score(y_true_d_np, y_pred_d_np, zero_division=0)
            f1 = f1_score(y_true_d_np, y_pred_d_np, zero_division=0)

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            model_3d = model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            recon_loss_per_sample = torch.mean(torch.pow((G_vae - model_3d), 2), dim=(1, 2, 3, 4))
            batch_recon_loss_avg_per_sample = torch.mean(recon_loss_per_sample)
            recon_loss_sum = torch.sum(torch.pow((G_vae - model_3d), 2))

            kl_loss = 0
            for i in range(args.num_views):
                kl_loss += KLLoss(z_mus[i], z_vars[i])

            batch_kl_loss_avg_per_sample = kl_loss / args.num_views / args.batch_size
            total_kl_loss_for_E = kl_loss / args.num_views
            E_loss = recon_loss_sum + total_kl_loss_for_E
            # print(f"rec los = {recon_loss_sum}")
            # print(f"kl los = {total_kl_loss_for_E*20}")
            E.zero_grad()
            E_loss.backward()
            E_solver.step()

            batch_iou = calculate_iou(G_vae, model_3d)
            
            try:
                batch_cd = calculate_chamfer_distance(G_vae, model_3d)
                if batch_cd != float('inf'):
                    epoch_chamfer_distance += batch_cd
                    num_cd_valid_batches += 1
            except Exception as e:
                print(f"Batch {num_batches} CD calculation error: {e}")

            Z = generateZ(args)
            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake.squeeze(), real_labels)
            # print(f"g_loss1= {g_loss*100}")
            Z_vae_detached = Z_vae.detach()
            G_vae_new = G(Z_vae_detached)
            recon_loss_new_sum = torch.sum(torch.pow((G_vae_new - model_3d),2))
            # print(f"g_loss2= {recon_loss_new_sum}")
            g_loss += recon_loss_new_sum

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            epoch_d_real_loss += d_real_loss.item()
            epoch_d_fake_loss += d_fake_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_acu += d_total_acu.item()
            epoch_recon_loss += batch_recon_loss_avg_per_sample.item() * args.batch_size
            epoch_kl_loss += batch_kl_loss_avg_per_sample.item() * args.batch_size
            epoch_d_precision += precision
            epoch_d_recall += recall
            epoch_d_f1 += f1
            epoch_iou += batch_iou
            num_batches += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        avg_iou = epoch_iou / num_batches
        avg_d_precision = epoch_d_precision / num_batches
        avg_d_recall = epoch_d_recall / num_batches
        avg_d_f1 = epoch_d_f1 / num_batches
        avg_d_real_loss = epoch_d_real_loss / num_batches
        avg_d_fake_loss = epoch_d_fake_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_acu = epoch_d_acu / num_batches
        avg_recon_loss = epoch_recon_loss / (num_batches * args.batch_size)
        avg_kl_loss = epoch_kl_loss / (num_batches * args.batch_size)
        avg_cd = epoch_chamfer_distance / num_cd_valid_batches if num_cd_valid_batches > 0 else float('inf')

        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)
            info = {
                'loss/loss_D_R': avg_d_real_loss, 'loss/loss_D_F': avg_d_fake_loss,
                'loss/loss_D': avg_d_loss, 'loss/loss_G': avg_g_loss, 'loss/acc_D': avg_d_acu,
                'loss/loss_recon': avg_recon_loss, 'loss/loss_kl': avg_kl_loss,
                'metric/iou': avg_iou, 'metric/chamfer_distance': avg_cd,
                'metric/d_precision': avg_d_precision, 'metric/d_recall': avg_d_recall, 'metric/d_f1': avg_d_f1,
                'time/epoch_duration_sec': epoch_duration
            }
            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)
            summary_writer.flush()

        cd_info = f", CD: {avg_cd:.6f}" if avg_cd != float('inf') else ", CD: N/A"
        print(
            f'Epoch-{epoch}; Time: {epoch_duration:.2f}s, IoU: {avg_iou:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}{cd_info}, D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, D_acu: {avg_d_acu:.4f}, D_Prec: {avg_d_precision:.4f}, D_Rec: {avg_d_recall:.4f}, D_F1: {avg_d_f1:.4f}, D_lr: {D_solver.state_dict()["param_groups"][0]["lr"]:.4f}')        

        if (epoch) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, epoch, G, G_solver, D, D_solver, E, E_solver)
