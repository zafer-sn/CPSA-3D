import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from PIL import Image
from torchvision import transforms
import binvox_rw
from augmentations import SilhouetteAugmentation

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data

def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]

class ShapeNetMultiviewDataset(data.Dataset):
    def __init__(self, root, args, is_train=True):
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.is_train = is_train

        # Base transforms (applied to all)
        self.base_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*4, [0.5]*4)
        ])

        # Augmentation transform (applied only to RGB channels during training)
        if self.is_train and self.args.use_silhouette_augmentation:
            self.augmentation_transform = SilhouetteAugmentation(p=0.5, value=0)
        else:
            self.augmentation_transform = None

    def __getitem__(self, index):
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]
        model_2d_files = [name for name in self.listdir if name.startswith(model_3d_file[:-7]) and name.endswith(".png")][:self.args.num_views]
        
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        
        images = []
        for x in model_2d_files:
            image = Image.open(self.root + x)
            # Apply base transforms to get the 4-channel tensor
            transformed_image = self.base_transforms(image)

            # If training and augmentation is enabled, apply it to RGB channels only
            if self.is_train and self.augmentation_transform is not None:
                # Separate RGB and Alpha channels
                rgb_channels = transformed_image[:3, :, :]
                alpha_channel = transformed_image[3:, :, :]

                # Apply augmentation to RGB channels
                rgb_channels_aug = self.augmentation_transform(rgb_channels)

                # Recombine the augmented RGB with the original Alpha
                transformed_image = torch.cat([rgb_channels_aug, alpha_channel], dim=0)
            
            images.append(transformed_image.clone())

        # Global index'i de döndür
        return (images, torch.FloatTensor(volume), index)

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return Variable(x)

def generateZ(args):
    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")
    return Z

########################## Pickle helper ###############################

def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:
        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files if file.startswith('G_') and file.endswith('.pkl')]
        file_list.sort()
        if not file_list:
            print("No pickle files found.")
            return 0
        recent_iter = str(file_list[-1])
        print(f"Continuing from iteration {recent_iter}")
        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f: G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f: G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f: D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f: D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f: E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f: E_solver.load_state_dict(torch.load(f))
        if os.path.exists(path + "/epoch_info_" + recent_iter + ".pkl"):
            with open(path + "/epoch_info_" + recent_iter + ".pkl", "rb") as f:
                epoch_info = torch.load(f)
                last_epoch = epoch_info.get('epoch', int(recent_iter))
                print(f"Continuing from epoch {last_epoch}")
                return last_epoch
        return int(recent_iter)
    except Exception as e:
        print("Pickle dosyaları okunurken hata:", e)
        return 0

def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None, E_solver=None):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f: torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f: torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f: torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f: torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f: torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f: torch.save(E_solver.state_dict(), f)
    with open(path + "/epoch_info_" + str(iteration) + ".pkl", "wb") as f: torch.save({'epoch': iteration}, f)

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred >= threshold).float()
    target = (target >= threshold).float()
    intersection = torch.sum(pred * target, dim=(1, 2, 3, 4))
    union = torch.sum(pred, dim=(1, 2, 3, 4)) + torch.sum(target, dim=(1, 2, 3, 4)) - intersection
    iou_per_sample = intersection / (union + 1e-6)
    return torch.mean(iou_per_sample).item()

def save_comparison_plot(input_image, generated_voxels, true_voxels, path, iteration, threshold=0.5, elev=30, azim=45):
    img_np = input_image.detach().cpu().numpy()
    if img_np.shape[0] in [3, 4]:
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    # Değer aralığını doğru ölçekle ([-1,1] -> [0,1], [0,255] -> [0,1], [0,1] -> [0,1])
    if img_np.max() > 1.0 and img_np.max() <= 255.0:
        img_np = img_np / 255.0
    elif img_np.min() < 0.0 and img_np.max() <= 1.0:
        img_np = (img_np + 1.0) / 2.0
    elif img_np.min() < 0.0 and img_np.max() > 1.0:
        # Aykırı durum: önce [-1,1] varsay, sonra kliple
        img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    gen_vox_np = (generated_voxels.detach().cpu().numpy().squeeze() >= threshold)
    true_vox_np = (true_voxels.detach().cpu().numpy().squeeze() >= threshold)
    gen_vox_np = np.transpose(gen_vox_np, (0, 2, 1))
    true_vox_np = np.transpose(true_vox_np, (0, 2, 1))
    intersection_vox_np = np.logical_and(gen_vox_np, true_vox_np)
    color_generated = '#FF3333'
    color_true = '#3333FF'
    color_intersection = '#33AA33'
    edge_color = '#555555'
    alpha_solid = 1.0
    alpha_overlay = 0.8
    def setup_3d_axes_common(ax, with_labels=True, title=''):
        dims = gen_vox_np.shape
        max_dim = max(dims)
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        if with_labels:
            if title:
                ax.set_title(title, color='black', fontsize=14, fontweight='bold')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_pane_color((0.95, 0.95, 0.98, 1.0))
        ax.yaxis.set_pane_color((0.93, 0.93, 0.96, 1.0))
        ax.zaxis.set_pane_color((0.91, 0.91, 0.94, 1.0))
        ax.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
        ax.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
        ax.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
    def create_labeled_figure():
        plt.style.use('default')
        fig = plt.figure(figsize=(22, 7))  # constrained_layout çıkarıldı
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.8, 1, 1, 1.2], figure=fig)
        gs.update(wspace=0.15, hspace=0)
        fig.patch.set_facecolor('white')
        ax1 = plt.subplot(gs[0])
        ax1.imshow(img_np)
        ax1.set_title('Input View', color='black', fontsize=14, fontweight='bold')
        ax1.set_facecolor('#F8F8F8')
        for spine in ax1.spines.values():
            spine.set_color('#CCCCCC')
        ax1.axis('off')
        ax2 = plt.subplot(gs[1], projection='3d')
        ax2.voxels(gen_vox_np, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax2, title='Generated Model')
        ax3 = plt.subplot(gs[2], projection='3d')
        ax3.voxels(true_vox_np, facecolors=color_true, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax3, title='Ground Truth')
        ax4 = plt.subplot(gs[3], projection='3d')
        true_only = np.logical_and(true_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(true_only, facecolors=color_true, edgecolor=edge_color, alpha=alpha_overlay)
        gen_only = np.logical_and(gen_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(gen_only, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_overlay)
        ax4.voxels(intersection_vox_np, facecolors=color_intersection, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax4)
        ax4.set_title('Overlay Comparison\nGreen: Match  Blue: GT Only  Red: Gen Only', 
                    color='black', fontsize=14, fontweight='bold')
        iou = np.sum(intersection_vox_np) / (np.sum(gen_vox_np) + np.sum(true_vox_np) - np.sum(intersection_vox_np))
        ax4.text2D(0.05, 0.05, f"IoU: {iou:.3f}", transform=ax4.transAxes, 
                color='black', fontsize=12, bbox=dict(facecolor='white', edgecolor='#AAAAAA', alpha=0.9))
        # tight_layout kaldırıldı; constrained_layout aktif
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.15)
        return fig
    fig_labeled = create_labeled_figure()
    fig_labeled.savefig(path + '/comparison_{}.png'.format(str(iteration).zfill(3)),
                        bbox_inches='tight', dpi=300, facecolor='white')
    fig_labeled.savefig(path + '/comparison_{}.pdf'.format(str(iteration).zfill(3)),
                        bbox_inches='tight', format='pdf', facecolor='white')
    plt.close(fig_labeled)

def voxel_to_obj(voxel_array, threshold=0.5, scale=1.0):
    voxels = (voxel_array >= threshold)
    vertices = []
    faces = []
    vertex_count = 0
    cube_corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]
    cube_faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z, y, x]:
                    for corner in cube_corners:
                        vertices.append([(x + corner[0]) * scale, (y + corner[1]) * scale, (z + corner[2]) * scale])
                    for face in cube_faces:
                        faces.append([vertex_count + face[0] + 1, vertex_count + face[1] + 1, vertex_count + face[2] + 1, vertex_count + face[3] + 1])
                    vertex_count += 8
    obj_content = "# Voxel model converted to OBJ\n"
    obj_content += "# Vertices: {}\n".format(len(vertices))
    obj_content += "# Faces: {}\n\n".format(len(faces))
    for vertex in vertices:
        obj_content += "v {} {} {}\n".format(vertex[0], vertex[1], vertex[2])
    for face in faces:
        obj_content += "f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])
    return obj_content

def save_image_copy(image_tensor, save_path, filename):
    """
    Save a copy of an image tensor to the specified path.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_np = image_tensor.detach().cpu().numpy()
    if img_np.shape[0] in [3, 4]:
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    # Normalize geri dönüş: [-1,1] -> [0,255], [0,1] -> [0,255], [0,255] -> [0,255]
    if img_np.max() <= 1.0 and img_np.min() >= -1.0:
        # Eğer [-1,1] ise
        if img_np.min() < 0.0:
            img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np * 255.0, 0, 255)
    elif img_np.max() > 1.0 and img_np.max() <= 255.0:
        img_np = np.clip(img_np, 0, 255)
    else:
        # Güvenli dönüşüm
        img_np = np.clip(img_np, 0, 1) * 255.0
    img_np = img_np.astype(np.uint8)
    from PIL import Image
    Image.fromarray(img_np).save(os.path.join(save_path, filename))

def voxel_to_points(voxel_array, threshold=0.5):
    import numpy as np
    binary_voxels = (voxel_array >= threshold)
    padded = np.pad(binary_voxels, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    points = []
    for z in range(1, padded.shape[0] - 1):
        for y in range(1, padded.shape[1] - 1):
            for x in range(1, padded.shape[2] - 1):
                if padded[z, y, x]:
                    neighbors = [padded[z-1, y, x], padded[z+1, y, x], padded[z, y-1, x], padded[z, y+1, x], padded[z, y, x-1], padded[z, y, x+1]]
                    if not all(neighbors):
                        points.append([x-1, y-1, z-1])
    return np.array(points).astype(np.float32)

def chamfer_distance_numpy(points1, points2):
    import numpy as np
    from scipy.spatial import distance_matrix
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    dist_matrix = distance_matrix(points1, points2)
    min_dist_1to2 = np.min(dist_matrix, axis=1)
    min_dist_2to1 = np.min(dist_matrix, axis=0)
    chamfer_dist = np.mean(min_dist_1to2) + np.mean(min_dist_2to1)
    return chamfer_dist

def calculate_chamfer_distance_gpu(pred, target, threshold=0.5, samples=1000):
    batch_size = pred.shape[0]
    cd_scores = []
    device = pred.device
    for i in range(batch_size):
        pred_binary = (pred[i, 0] >= threshold)
        target_binary = (target[i, 0] >= threshold)
        pred_indices = torch.nonzero(pred_binary, as_tuple=False).float()
        target_indices = torch.nonzero(target_binary, as_tuple=False).float()
        if len(pred_indices) == 0 or len(target_indices) == 0:
            continue
        if len(pred_indices) > samples:
            perm = torch.randperm(len(pred_indices), device=device)
            pred_indices = pred_indices[perm[:samples]]
        if len(target_indices) > samples:
            perm = torch.randperm(len(target_indices), device=device)
            target_indices = target_indices[perm[:samples]]
        pred_squared = torch.sum(pred_indices**2, dim=1, keepdim=True)
        target_squared = torch.sum(target_indices**2, dim=1, keepdim=True)
        squared_dist = pred_squared + target_squared.t() - 2 * torch.mm(pred_indices, target_indices.t())
        dist_pred_to_target, _ = torch.min(squared_dist, dim=1)
        dist_target_to_pred, _ = torch.min(squared_dist, dim=0)
        cd = torch.mean(torch.sqrt(dist_pred_to_target)) + torch.mean(torch.sqrt(dist_target_to_pred))
        cd_scores.append(cd.item())
    if len(cd_scores) > 0:
        return sum(cd_scores) / len(cd_scores)
    else:
        return float('inf')

def calculate_chamfer_distance(pred, target, threshold=0.5, samples=1000):
    if pred.is_cuda and target.is_cuda:
        return calculate_chamfer_distance_gpu(pred, target, threshold, samples)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch_size = pred.shape[0]
    cd_scores = []
    for i in range(batch_size):
        pred_points = voxel_to_points(pred_np[i, 0], threshold)
        target_points = voxel_to_points(target_np[i, 0], threshold)
        if len(pred_points) == 0 or len(target_points) == 0:
            continue
        if len(pred_points) > samples:
            indices = np.random.choice(len(pred_points), samples, replace=False)
            pred_points = pred_points[indices]
        if len(target_points) > samples:
            indices = np.random.choice(len(target_points), samples, replace=False)
            target_points = target_points[indices]
        cd = chamfer_distance_numpy(pred_points, target_points)
        cd_scores.append(cd)
    if len(cd_scores) > 0:
        return np.mean(cd_scores)
    else:
        return float('inf')
