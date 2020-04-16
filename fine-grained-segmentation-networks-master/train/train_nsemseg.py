import datetime
import os
import sys
import numpy as np
import h5py
import copy
from math import sqrt
import torch
import faiss
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import glob

import context

import datasets.dataset_configs as data_configs
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import utils.corr_transforms as corr_transforms

from datasets import correspondences, merged , Poladata
from models import model_configs
from utils.misc import check_mkdir, AverageMeter, freeze_bn, get_global_opts, rename_keys_to_match, get_latest_network_name, clean_log_before_continuing, load_resnet101_weights, get_network_name_from_iteration
from utils.validator import CorrValidator
from layers.feature_loss import FeatureLoss
from layers.cluster_correspondence_loss import ClusterCorrespondenceLoss

from clustering import clustering
from clustering.cluster_tools import extract_features_for_reference_nocorr, extract_features_for_reference, save_cluster_features_as_segmentations
from clustering.clustering import preprocess_features


def init_last_layers(state_dict, n_clusters):
    device = state_dict['conv6.weight'].device
    state_dict['conv6.weight'] = torch.zeros([n_clusters, state_dict['conv6.weight'].size(
        1), state_dict['conv6.weight'].size(2), state_dict['conv6.weight'].size(3)],
        dtype=torch.float, device=device)
    state_dict['conv6.bias'] = torch.zeros(
        [n_clusters], dtype=torch.float, device=device)
    state_dict['conv6_1.weight'] = torch.zeros([n_clusters, state_dict['conv6_1.weight'].size(1),
                                                state_dict['conv6_1.weight'].size(
                                                    2),
                                                state_dict['conv6_1.weight'].size(3)],
                                               dtype=torch.float, device=device)
    state_dict['conv6_1.bias'] = torch.zeros(
        [n_clusters], dtype=torch.float, device=device)


def reinit_last_layers(net):
    net.conv6.weight.data.normal_(0, 0.01)
    net.conv6_1.weight.data.normal_(0, 0.01)
    net.conv6.bias.data.zero_()
    net.conv6_1.bias.data.zero_()


def train_with_clustering(save_folder, tmp_seg_folder, startnet, args):
    print(save_folder.split('/')[-1])
    skip_clustering = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    check_mkdir(save_folder)
    writer = SummaryWriter(save_folder)
    check_mkdir(tmp_seg_folder)

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network(n_classes=args['n_clusters'], for_clustering=True,
                                    output_features=True, use_original_base=args['use_original_base']).to(device)

    state_dict = torch.load(startnet)
    if 'resnet101' in startnet:
        load_resnet101_weights(net, state_dict)
    else:
        # needed since we slightly changed the structure of the network in pspnet
        state_dict = rename_keys_to_match(state_dict)
        # different amount of classes
        init_last_layers(state_dict, args['n_clusters'])

        net.load_state_dict(state_dict)  # load original weights

    start_iter = 0
    args['best_record'] = {'iter': 0, 'val_loss_feat': 1e10,
                           'val_loss_out': 1e10, 'val_loss_cluster': 1e10}

    # Data loading setup
    if args['corr_set'] == 'rc':
        corr_set_config = data_configs.RobotcarConfig()
    elif args['corr_set'] == 'pola':
        corr_set_config = data_configs.PolaConfig()
    elif args['corr_set'] == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif args['corr_set'] == 'both':
        corr_set_config1 = data_configs.CmuConfig()
        corr_set_config2 = data_configs.RobotcarConfig()

    ref_image_lists = [corr_set_config.reference_image_list]

    # ref_image_lists = glob.glob("/media/HDD1/datasets/Creusot_Jan15/Creusot_3/*.jpg", recursive=True)
    # print(f'ici on print ref image list ---------------------------------------------------- {ref_image_lists}')
    # print(corr_set_config)
    # corr_im_paths = [corr_set_config.correspondence_im_path]
    # ref_featurs_pos = [corr_set_config.reference_feature_poitions]

    input_transform = model_config.input_transform

    #corr_set_train = correspondences.Correspondences(corr_set_config.correspondence_path,
    #                                                 corr_set_config.correspondence_im_path,
    #                                                 input_size=(713, 713),
    #                                                 input_transform=input_transform,
    #                                                 joint_transform=train_joint_transform_corr,
    #                                                 listfile=corr_set_config.correspondence_train_list_file)
    scales = [0,1,2,3]

    # corr_set_train = Poladata.MonoDataset(corr_set_config,
    #                                       seg_folder = "media/HDD1/NsemSEG/Result_fold/" ,
    #                                       im_file_ending = ".jpg" )
    corr_set_train = Poladata.MonoDataset(corr_set_config.train_im_folder,
                                          corr_set_config.train_seg_folder,
                                          im_file_ending = ".jpg",
                                          # input_transform = input_transform,
                                          id_to_trainid = None,
                                          joint_transform = None,
                                          sliding_crop = None,
                                          transform = None,
                                          target_transform = None,
                                          transform_before_sliding = None
                                          )
    # print (corr_set_train)
    # print(corr_set_train.mask)
    corr_loader_train = DataLoader(
        corr_set_train, batch_size=1, num_workers=args['n_workers'], shuffle=True)

    # print(corr_loader_train)
    seg_loss_fct = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    # Clustering
    deepcluster = clustering.Kmeans(args['n_clusters'])
    if skip_clustering:
        deepcluster.set_index(cluster_centroids)

    open(os.path.join(save_folder, str(datetime.datetime.now()) + '.txt'),
         'w').write(str(args) + '\n\n')

    f_handle = open(os.path.join(save_folder, 'log.log'), 'w', buffering=1)

    # clean_log_before_continuing(os.path.join(save_folder, 'log.log'), start_iter)
    # f_handle = open(os.path.join(save_folder, 'log.log'), 'a', buffering=1)

    val_iter = 0
    curr_iter = start_iter
    while curr_iter <= args['max_iter']:

        net.eval()
        net.output_features = True
        # max_num_features_per_image = args['max_features_per_image']
        # print('-----------------------------------------------------------------')
        # print (f'ref_image_lists est: {ref_image_lists},model_config es : {model_config} , net es: {net} , max feature par image es : {max_num_features_per_image} ')
        # print('-----------------------------------------------------------------')

        features = extract_features_for_reference_nocorr(net, model_config, ref_image_lists,
                                                    len(ref_image_lists),
                                                    max_num_features_per_image=args['max_features_per_image'])

        cluster_features = np.vstack(features)
        del features

        # cluster the features
        cluster_indices, clustering_loss, cluster_centroids, pca_info = deepcluster.cluster_imfeatures(
            cluster_features, verbose=True, use_gpu=False)

        # save cluster centroids
        h5f = h5py.File(os.path.join(
            save_folder, 'centroids_%d.h5' % curr_iter), 'w')
        h5f.create_dataset('cluster_centroids', data=cluster_centroids)
        h5f.create_dataset('pca_transform_Amat', data=pca_info[0])
        h5f.create_dataset('pca_transform_bvec', data=pca_info[1])
        h5f.close()

        # Print distribution of clusters
        cluster_distribution, _ = np.histogram(
            cluster_indices, bins=np.arange(args['n_clusters'] + 1), density=True)
        str2write = 'cluster distribution ' + \
            np.array2string(cluster_distribution, formatter={
                            'float_kind': '{0:.8f}'.format}).replace('\n', ' ')
        print(str2write)
        f_handle.write(str2write + "\n")

        # set last layer weight to a normal distribution
        reinit_last_layers(net)

        # make a copy of current network state to do cluster assignment
        net_for_clustering = copy.deepcopy(net)

        optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                            ) ** args['lr_decay']
        optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                        ) ** args['lr_decay']

        net.train()
        freeze_bn(net)
        net.output_features = False
        cluster_training_count = 0

        # Train using the training correspondence set
        corr_train_loss = AverageMeter()
        seg_train_loss = AverageMeter()
        feature_train_loss = AverageMeter()

        while cluster_training_count < args['cluster_interval'] and curr_iter <= args['max_iter']:

            # First extract cluster labels using saved network checkpoint
            net.to("cpu")
            net_for_clustering.to(device)
            net_for_clustering.eval()
            net_for_clustering.output_features = True

            data_samples = []
            extract_label_count = 0
            while (extract_label_count < args['chunk_size']) and (cluster_training_count + extract_label_count < args['cluster_interval']) and (val_iter + extract_label_count < args['val_interval']) and (extract_label_count + curr_iter <= args['max_iter']):
                img_ref, img_other, pts_ref, pts_other, _ = next(
                    iter(corr_loader_train))

                print(img_ref)

                # Transfer data to device
                img_ref = img_ref.to(device)

                with torch.no_grad():
                    features = net_for_clustering(img_ref)

                # assign feature to clusters for entire patch
                output = features.cpu().numpy()
                output_flat = output.reshape(
                    (output.shape[0], output.shape[1], -1))
                cluster_image = np.zeros(
                    (output.shape[0], output.shape[2], output.shape[3]), dtype=np.int64)
                for b in range(output_flat.shape[0]):
                    out_f = output_flat[b]
                    out_f2, _ = preprocess_features(
                        np.swapaxes(out_f, 0, 1), pca_info=pca_info)
                    cluster_labels = deepcluster.assign(out_f2)
                    cluster_image[b] = cluster_labels.reshape(
                        (output.shape[2], output.shape[3]))

                cluster_image = torch.from_numpy(cluster_image).to(device)

                # assign cluster to correspondence positions
                cluster_labels = assign_cluster_ids_to_correspondence_points(
                    features, pts_ref, (deepcluster, pca_info), inds_other=pts_other, orig_im_size=(713, 713))

                # Transfer data to cpu
                img_ref = img_ref.cpu()
                cluster_labels = [p.cpu() for p in cluster_labels]
                cluster_image = cluster_image.cpu()
                data_samples.append((img_ref, cluster_labels, cluster_image))
                extract_label_count += 1

            net_for_clustering.to("cpu")
            net.to(device)

            for data_sample in data_samples:
                img_ref, cluster_labels, cluster_image = data_sample

                # Transfer data to device
                img_ref = img_ref.to(device)
                cluster_labels = [p.to(device) for p in cluster_labels]
                cluster_image = cluster_image.to(device)

                optimizer.zero_grad()

                outputs_ref, aux_ref = net(img_ref)

                seg_main_loss = seg_loss_fct(outputs_ref, cluster_image)
                seg_aux_loss = seg_loss_fct(aux_ref, cluster_image)

                loss = args['seg_loss_weight'] * \
                    (seg_main_loss + 0.4 * seg_aux_loss)

                loss.backward()
                optimizer.step()
                cluster_training_count += 1

                if type(seg_main_loss) == torch.Tensor:
                    seg_train_loss.update(seg_main_loss.item(), 1)

                ####################################################################################################
                #       LOGGING ETC
                ####################################################################################################
                curr_iter += 1
                val_iter += 1

                writer.add_scalar('train_seg_loss',
                                  seg_train_loss.avg, curr_iter)
                writer.add_scalar(
                    'lr', optimizer.param_groups[1]['lr'], curr_iter)

                if (curr_iter + 1) % args['print_freq'] == 0:
                    str2write = '[iter %d / %d], [train seg loss %.5f], [train corr loss %.5f], [train feature loss %.5f]. [lr %.10f]' % (
                        curr_iter + 1, args['max_iter'], seg_train_loss.avg, optimizer.param_groups[1]['lr'])

                    print(str2write)
                    f_handle.write(str2write + "\n")

                if curr_iter > args['max_iter']:
                    break

    # Post training
    f_handle.close()
    writer.close()


def generate_name_of_result_folder(args):
    global_opts = get_global_opts()

    results_path = os.path.join(global_opts['result_path'], 'cluster-training')
    if 'vis' == args['startnet']:
        startnetstr = 'map1'
    elif 'cs' == args['startnet']:
        startnetstr = 'map0'
    elif 'pola' == args['startnet']:
        startnetstr = 'map2'
    else:
        startnetstr = 'other'

    cluster_str = 'features%d' % (args['max_features_per_image'])

    if args['feature_hinge_loss_weight'] == 0:
        result_folder = 'cluster-%s-%s-cn%d-ci%d-vi%d' % (
            args['corr_set'], startnetstr, args['n_clusters'], args['cluster_interval'],
            args['val_interval'])
    else:
        result_folder = 'cluster-%s-%s-cn%d-ci%d-vi%d-ws%.5f-wf%.5f-%s-valm' % (
            args['corr_set'], startnetstr, args['n_clusters'], args['cluster_interval'],
            args['val_interval'],args['seg_loss_weight'],
            args['feature_hinge_loss_weight'], cluster_str)

    return os.path.join(results_path, result_folder), os.path.join(global_opts['result_path'], result_folder)


def get_path_of_startnet(args):
    global_opts = get_global_opts()

    if args['startnet'] == 'vis':
        return os.path.join(global_opts['result_path'], 'base-networks', 'pspnet101_cs_vis.pth')
    elif args['startnet'] == 'cs':
        return os.path.join(global_opts['result_path'], 'base-networks', 'pspnet101_cityscapes.pth')
    elif args['startnet'] == 'pola':
        return os.path.join(global_opts['models_path'],'base-networks', 'pspnet101_cityscapes.pth')


def train_with_clustering_experiment(args):
    if args['startnet'] in ['vis', 'cs' ,'pola']:
        startnet = get_path_of_startnet(args)
    else:
        startnet = args['startnet']

    save_folder, tmp_folder = generate_name_of_result_folder(args)
    train_with_clustering(save_folder, tmp_folder, startnet, args)


if __name__ == '__main__':
    args = {
        # general training settings
        'train_batch_size': 1,
        # probability of propagating error for reference image instead of target imare (set to None to use both)
        'fraction_reference_bp': 0.5,
        'lr': 1e-4 / sqrt(16 / 1),
        'lr_decay': 1,
        'max_iter': 60000,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # starting network settings
        'startnet': 'pola',  # specify full path or set to 'vis' for network trained with vistas + cityscapes or 'cs' for network trained with cityscapes
        'use_original_base': False,  # must be true if starting from classification network

        # set to '' to start training from beginning and 'latest' to use last checkpoint
        'snapshot': '',

        # dataset settings
        'corr_set': 'pola',  # 'cmu', 'rc', 'both' or 'none'
        'max_features_per_image': 500,  # dont set to high (RAM runs out)

        # clustering settings
        'n_clusters': 100,
        'cluster_interval': 10000,

        # loss settings
        'corr_loss_weight': 1,  # was 1
        'seg_loss_weight': 1,  # was 1
        'feature_hinge_loss_weight': 0,  # was 0

        # validation settings
        'val_interval': 2500,
        'feature_distance_measure': 'L2',

        # misc
        'chunk_size': 50,
        'print_freq': 10,
        'stride_rate': 2 / 3.,
        'n_workers': 1,  # set to 0 for debugging
    }
    train_with_clustering_experiment(args)



#
# from sklearn.model_selection import train_test_split
# import sys
# from os import walk
# import os
# import glob
#
# assert(len(sys.argv) > 2)
#
# folder = sys.argv[1]
# split_name = sys.argv[2]
# folders = [x[0] for x in os.walk(folder)]
# folders.pop(0)
# images = []
# for f in folders:
#     fol = list(dict.fromkeys(glob.iglob(os.path.join(f, "*.jpg"))))
#     if 'rain' in f:
#         lentoremove = len(fol)
#     else:
#         lentoremove = len(fol) - 1
#
#     removed = os.path.join(f, str(lentoremove).zfill(5) + '.jpg')
#     fol.remove(removed)
#     if 'rain' in f:
#         removed = os.path.join(f, str(1).zfill(5) + '.jpg')
#         fol.remove(removed)
#     else:
#         removed = os.path.join(f, str(0).zfill(5) + '.jpg')
#         fol.remove(removed)
#
#     images += fol
#
# # for f in folders:
# #     images.append('test')
# # f = glob.glob(folder + '/**/*.jpg', recursive=True)
# splits = train_test_split(images, test_size=0.1, random_state=0)
# qualifier = ['train', 'val']
#
# for idx, lst in enumerate(splits):
#     fileout = f'{split_name}/{qualifier[idx]}_files.txt'
#     with open(fileout, 'w') as outfile:
#         for img in lst:
#             img_ = img.split('/')[-1]
#             index = str(int(img_.split('.')[0]))
#             if not int(index) == 0:
#                 firstpart = img.split('/')[-2]
#                 to_append = f'{firstpart} {index} l\n'
#                 outfile.write(to_append)
#
# # print(len(f))
# # print(len(out[0]))
# # print(len(out[1]))

