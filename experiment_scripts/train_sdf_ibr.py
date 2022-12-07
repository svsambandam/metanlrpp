"""
Trains the NLR++ model.
"""
# Enable import from parent package
from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

# Import utils early to initialize matplotlib.
import utils.common_utils as common_utils

import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import configargparse

import data_processing.datasets.dataio_sdf as dataio_sdf
from utils.ray_builder import RayBuilder
import utils.utils_ibr as utils_ibr
import training
import loss_functions
import modules_sdf

import time
start_time = time.time()


def get_arg_parser():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    # Save/resume.
    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                   help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help='Checkpoint to trained model. Latest used as default.')
    p.add_argument('--checkpoint_strict', type=int, default=1,
                   help='Is the checkpoint strict (containing all modules)?')
    p.add_argument('--checkpoint_sdf', type=str, default=None,
                   help='Checkpoint to only use for SDF. Overrides defaults.')
    p.add_argument('--checkpoint_img_encoder', type=str, default=None,
                   help='Checkpoint to only use for Image Encoder.')
    p.add_argument('--checkpoint_img_decoder', type=str, default=None,
                   help='Checkpoint to only use for Image Decoder.')
    p.add_argument('--checkpoint_aggregation_mlp', type=str, default=None,
                   help='Checkpoint to only use for Aggregation MLP.')
    p.add_argument('--resume', type=int, default=1,
                   help='Resume from previous checkpoint?')
    p.add_argument('--restart', type=int, default=1,
                   help='Remove all prev checkpoints and summaries in the log dir?')
    p.add_argument('--verbose_logging', type=int, default=0,
                   help='Save complete state every single iteration.')
    p.add_argument('--load_verbose_record', type=str, default=None,
                   help='Loads verbose record for debugging.')
    p.add_argument('--load_model_poses', type=int, default=0,
                   help='Load model poses from the trained SDF network. This must be set when there is a '
                        'dataset mismatch')

    # General training options
    p.add_argument('--device', type=str, default='cuda', help='Device to use.')
    p.add_argument('--sphere_tracing_iters', type=int, default=10, help='Number of Sphere tracing steps. Set to 0 if want to skip this and go to Secant.')
    p.add_argument('--batch_size', type=int, default=32768, help='Number of points for 3D supervision') #32768
    p.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs to train for.')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--lr_sdf', type=float, default=5e-5, help='learning rate for sdf. default=5e-5.')
    p.add_argument('--lr_decay_factor', type=float, default=0.5, help='How to much to decay LR.')
    p.add_argument('--lr_sdf_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_encdec_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_agg_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_alternating_interval', type=int, default=0,
                   help='How often (steps) to swap color and sdf training.')

    p.add_argument('--epochs_til_ckpt', type=int, default=1,
                   help='Time interval in epochs until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=1,
                   help='Time interval in steps until tensorboard summary is saved.')
    p.add_argument('--image_batch_size', type=int, default=4, help='Number of target images per batch.')

    # Implicit Models
    p.add_argument('--model', type=str, default='ours',
                   help='Predefined models [ours|idr|deepsdf]')
    p.add_argument('--model_activation_sdf', type=str, default='sine',
                   help='Activation function [sine|relu]')
    p.add_argument('--model_hidden_layers_sdf', type=int, default=3,
                   help='How many hidden layers between 1st and last.')
    p.add_argument('--model_hidden_dims_sdf', type=int, default=128,
                   help='How many dimensions in hidden layers.')
    p.add_argument('--model_skips_sdf', type=str, default='none',
                   help='Comma separated skip connections.')
    p.add_argument('--feature_vector', type=int, default=-1,
                   help='IDR-like feature vector size.')

    # CNN Models
    p.add_argument('--model_image_encoder_depth', type=int, default=3,
                   help='Depth of the ResNet used for encoding individual images.')
    p.add_argument('--model_image_encoder_features', type=int, default=16,
                   help='Size of output features of each image.')
    p.add_argument('--model_image_decoder_depth', type=int, default=2,
                   help='Depth of the UNet used for decoding an image from feature.')
    p.add_argument('--feature_type', type=str, default='learned',
                   help='Type of features. Whether they are learned, or simply RGB values from input '
                        'views. [learned|rgb]')

    # Feature aggregation methods
    p.add_argument('--feature_aggregation_method', type=str, default='sum',
                   help='Feature aggregation method [sum|lumigraph(_epipolar)|mean|mlp].')
    p.add_argument('--source_views_per_target', type=int, default=25,
                   help='Number of source views to sample features from to render target.')
    p.add_argument('--total_number_source_views', type=int, default=-1,
                   help='Total number of source views to select from.')
    p.add_argument('--source_view_selection_mode', type=str, default='random',
                   help='Method for selecting source views to sample features from [random|nearest]')
    p.add_argument('--occlusion_method', type=str, default='raytrace',
                   help='Method for deciding if features are occluded or not')

    # Positional encoding.
    p.add_argument('--posenc_sdf', type=str, default='none',
                   help='Positional encoding for SDF [none|nerf|idr|ff].')
    p.add_argument('--posenc_warp_sdf', type=str, default='none',
                   help='Positional encoding for warp SDF [none|nerf|idr|ff].')
    p.add_argument('--posenc_warp_sdf_type', type=str, default='none',
                   help='Input typefor positional encoding for warp SDF [none|target_view_id|coords].')
    p.add_argument('--posenc_sdf_bands', type=int, default=0,
                   help='Number of pos enc bands.')
    p.add_argument('--posenc_sdf_sigma', type=float, default=1,
                   help='Sigma value for FF encoding.')
    p.add_argument('--posenc_sdf_expbands', type=int, default=0,
                   help='Use exponential band sequence for IDR encoding?')

    # SDF Network
    p.add_argument('--warping', type=str, default=None,
                   help='Use warping MLP for non-static scenes?')
    p.add_argument('--hyperwarp', type=int, default=0,
                   help='Use hyperwarp MLP for non-static scenes? 1 for yes, 0 for no')
    p.add_argument('--hyper_dim', type=int, default=0,
                   help='How many ambient dimensions used in hyperwarp MLP?')
    p.add_argument('--init_regularized', type=int, default=0,
                   help='Use regularized weights for the sphere init?')
    p.add_argument('--fit_sphere', type=int, default=0,
                   help='Should we train for sphere only? Used to create init weights.')
    p.add_argument('--init_sphere', type=int, default=0,
                   help='Should we initialize the weights to represent unit sphere?')
    p.add_argument('--fit_plane', type=int, default=0,
                   help='Should we train for plane only? Used to create init weights.')
    p.add_argument('--init_plane', type=int, default=0,
                   help='Should we initialize the weights to represent Z=0 plane?')

    # Dataset
    p.add_argument('--dataset_path', type=str, default='/home/data/',
                   help='Path to dataset folder.')
    p.add_argument('--dataset_type', type=str, default='sinesdf_static',
                   help='Dataset type [sinesdf_static].')
    p.add_argument('--dataset_name', type=str, default='dtu',
                   help='Dataset name [dtu|nlr|nerfies|shapenet]')
    p.add_argument('--world_pcd_path', type=str, default='',
                   help='Alternative path to PCD to use instead of the dataset.')
    p.add_argument('--load_pcd', type=int, default=1,
                   help='Should we load PCD for training or testing?')
    p.add_argument('--use_pcd', type=int, default=1,
                   help='Should we use PCD for training?')
    p.add_argument('--load_images', type=int, default=1,
                   help='Should we load images for training or testing?')
    p.add_argument('--work_radius', type=float, default=0.99,
                   help='To how large sphere to scale the model?')
    p.add_argument('--scene_radius_scale', type=float, default=1.0,
                   help='Scale the radius estimated from camera intersection?')
    p.add_argument('--scene_normalization', type=str, default='cache,yaml,pcd,camera',
                   help='Sets prefered space normalization mode order.')
    p.add_argument('--reference_view', type=int, default=-1,
                   help='Which view to use for RT preview? Use -1 for mid view.')
    p.add_argument('--test_views', type=str, default='',
                   help='Comma separated list of views to hold out for test purposes. Zero based indices.')
    p.add_argument('--randomize_cameras', type=int, default=0,
                   help='Should I add noise to camera poses?')

    p.add_argument('--load_im_scale', type=float, default=1.0,
                   help="Scale factor for the image training resolution. Changes the base of other scales.")
    p.add_argument('--im_scale', type=float, default=1.0,
                   help="Scale factor for the image render resolution. Only affects tests/eval/summaries.")
    p.add_argument("--color_loss", type=str, default='l2',
                   help='Which loss to use for color: l1|l2|smooth_l1')

    p.add_argument('--precomputed_3D_point_buffers', type=str, default=None,
                   help='Location of precomputed 3D position buffers.')
    p.add_argument('--save_3D_point_buffers', type=int, default=0,
                   help='Save the precomputed 3D position buffers for quicker load next run.')
    p.add_argument('--load_3D_point_buffers', type=int, default=0,
                   help='Load the precomputed 3D position buffers at directory.')

    # Ray-tracing.
    p.add_argument('--rt_bidirectional', type=int, default=1,
                   help='Use bidirectional ray tracing?.')
    p.add_argument('--rt_num_steps', type=int, default=16,
                   help='Number of steps for each ray.')
    p.add_argument('--rt_num_section_steps', type=int, default=100,
                   help='Number of uniform steps for sectioning.')
    p.add_argument('--rt_num_secant_steps', type=int, default=8,
                   help='Number of steps for secant algorithm.')
    p.add_argument('--rt_num_mask_steps', type=int, default=100,
                   help='Number of uniform steps for differentiable mask.')
    p.add_argument('--rt_step_alpha', type=float, default=1.0,
                   help="Ray step length factor.")
    p.add_argument('--rt_mask_alpha', type=float, default=50.0,
                   help="The mask softness alpha from Lipman 2020 Eq. 7.")
    p.add_argument('--rt_mask_alpha_period', type=int, default=5000,
                   help="Double the rt_mask_alpha every n steps.")
    p.add_argument('--rt_mask_alpha_period_epochs', type=int, default=-1,
                   help="Double the rt_mask_alpha every n epochs.")
    p.add_argument('--rt_mask_alpha_period_max', type=int, default=5,
                   help="Double the rt_mask_alpha at most this epochs times.")
    p.add_argument('--rt_mask_loss_weight', type=float, default=0.03,
                   help="Weight on the false positive rays mask loss.")

    # Parameters. Train which decoder?
    p.add_argument('--train_decoder_sdf', type=int, default=1,
                   help='Optimize SDF decoder?')
    p.add_argument('--train_image_encoder', type=int, default=1,
                   help='Optimize image encoder?')
    p.add_argument('--train_feature_decoder', type=int, default=1,
                   help='Optimize feature decoder?')
    p.add_argument('--train_feature_blending', type=int, default=1,
                   help='Optimize feature blender?')

    # Losses. They are always computed (if possible) but
    # can be left-out of the optimization.
    p.add_argument('--opt_sdf_onsurface', type=int, default=1,
                   help='Optimize On-surface SDF == 0')
    p.add_argument('--opt_sdf_offsurface', type=int, default=1,
                   help='Optimize Off-surface |SDF| > 0')
    p.add_argument('--opt_sdf_normal', type=int, default=1,
                   help='Optimize On-surface normal == GT')
    p.add_argument('--opt_sdf_eikonal', type=int, default=1,
                   help='Optimize ||Grad SDF|| = 1')
    p.add_argument('--opt_sdf_eikonal_w', type=float, default=1.0,
                   help='Weight for Optimize ||Grad SDF|| = 1')
    p.add_argument('--opt_sdf_front', type=int, default=0,
                   help='Should I use front constraint (for Kinect)? Forces positive SDF in front of mesh.')
    p.add_argument('--opt_sdf_rear', type=int, default=0,
                   help='Should I use front constraint (for Kinect)? Forces negative SDF behind the mesh.')
    p.add_argument('--opt_sdf_lapl', type=int, default=0,
                   help='Should I use laplacian constraint? Minimizes SDF laplacian.')
    p.add_argument('--opt_sdf_direct', type=int, default=0, help='Directly optimize SDF.')
    p.add_argument('--opt_sdf_curvature', type=float, default=0,
                   help='Minimize sdf curvature?')
    p.add_argument('--opt_rays_sdf_curvature', type=float, default=0,
                   help='Minimize sdf curvature on the surface?')

    # Additional loss options.
    p.add_argument('--loss_eikonal_metric', type=str, default='l1',
                   help='Metric used for the eikonal loss l1|l2')
    p.add_argument('--opt_render_shape', type=int, default=0,
                   help='Optimize SDF shape using the view image raytracing loss?')
    p.add_argument('--opt_render_softmask', type=int, default=0,
                   help='Optimize the contour mask?')
    p.add_argument('--regularize_weights_sdf', type=float, default=0.0,
                   help='Minimize weights in the MLP?')
    p.add_argument('--opt_perceptual_loss', type=int, default=0,
                   help='Use perceptual loss on rendered target images')

    return p


def get_sdf_decoder(opt, dataset: dataio_sdf.DatasetSDF = None):

    # Losses.
    if not opt.opt_sdf_eikonal:
        opt.opt_sdf_eikonal_w = 0.0

    # RayBuilder - only needed for 2D training.
    ray_builder = None
    if dataset is not None and dataset.dataset_img is not None:
        ray_builder = RayBuilder(opt, dataset.dataset_img, dataset.model_matrix)
    # print('---------------------trainsdfibr265------------------', len(dataset.dataset_img), dataset.model_matrix, ray_builder)

    # Define the model.
    model = modules_sdf.SDFIBRNet(opt, ray_builder=ray_builder)
    model.to(opt.device)
    return model


def get_dataset(opt, WITHHELD_VIEWS=None):
    """
    Gets the dataset.
    """
    if WITHHELD_VIEWS is None:
        if opt.dataset_name == 'dtu':
            # opt.TRAIN_VIEWS = [1, 9, 17, 25, 33, 41, 47]
            # opt.WITHHELD_VIEWS = list(set(list(range(0, 49))) - set(opt.TRAIN_VIEWS))
            # opt.TRAIN_VIEWS = [0, 6, 14, 20, 27, 34, 39] # my_dtu
            # opt.WITHHELD_VIEWS = list(set(list(range(0, 41))) - set(opt.TRAIN_VIEWS))
            opt.TRAIN_VIEWS = [1, 9, 17, 23, 31, 38, 44]
            opt.WITHHELD_VIEWS = list(set(list(range(0, 46))) - set(opt.TRAIN_VIEWS))
        elif opt.dataset_name == 'nlr':
            # opt.TRAIN_VIEWS = [16, 17, 18, 20, 21, 19]
            opt.TRAIN_VIEWS = [0, 1, 3, 4, 5]
            print('I CHNAGED NLR TRAINING VIEWS')
            # opt.WITHHELD_VIEWS = list(set(list(range(0, 21))) - set(opt.TRAIN_VIEWS))
            opt.WITHHELD_VIEWS = list(set(list(range(0, 6))) - set(opt.TRAIN_VIEWS))
        # elif opt.dataset_name == 'nlr':
        #     print('WARNING-------------------------------- this is for lv1 lowres!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     opt.TRAIN_VIEWS = [0, 1, 12, 3, 13, 7, ]#[0, 2, 4, 7, 9] #[0, 2, 3, 5, 7]
        #     print('I CHNAGED NLR TRAINING VIEWS')
        #     opt.WITHHELD_VIEWS = list(set(list(range(0, 15))) - set(opt.TRAIN_VIEWS))
        elif opt.dataset_name == 'curls':
            opt.TRAIN_VIEWS =  [0, 15, 26, 28, 30, 34, 44]
            opt.WITHHELD_VIEWS = list(set(list(range(0, 50))) - set(opt.TRAIN_VIEWS))
        elif opt.dataset_name == 'toby':
            raise(NotImplementedError)
            opt.TRAIN_VIEWS =  [0, 15, 26, 28, 30, 34, 44]
            opt.WITHHELD_VIEWS = list(set(list(range(0, 50))) - set(opt.TRAIN_VIEWS))
        elif opt.dataset_name == 'mynlr':
            opt.TRAIN_VIEWS =  [0, 12, 3, 13, 7 ]#[0, 2, 4, 7, 9]
            opt.WITHHELD_VIEWS = list(set(list(range(0, 15))) - set(opt.TRAIN_VIEWS))
        elif opt.dataset_name == 'shapenet':
            opt.WITHHELD_VIEWS = [7, 16, 23]
            opt.TRAIN_VIEWS = list(set(list(range(0, 23))) - set(opt.WITHHELD_VIEWS))
        print(f'Training Views: {opt.TRAIN_VIEWS}.')
    else:
        opt.WITHHELD_VIEWS = WITHHELD_VIEWS
    return dataio_sdf.DatasetSDF(Path(opt.dataset_path), opt)


def get_latest_checkpoint_file(opt):
    """
    Gets the latest checkpoint pth file.
    """
    chck_dir = Path(opt.logging_root) / opt.experiment_name / 'checkpoints'

    # Return final if exists.
    chck_final = chck_dir / 'model_final.pth'
    if chck_final.is_file():
        return chck_final

    # Return current if exists.
    chck_current = chck_dir / 'model_current.pth'
    if chck_current.is_file():
        return chck_current

    # Find all pth files.
    if chck_dir.is_dir():
        check_files = sorted([x for x in chck_dir.iterdir() if x.stem.startswith('model') and x.suffix == '.pth'])
    else:
        check_files = []
    if check_files:
        # Return latest.
        return check_files[-1]

    # No checkpoint.
    return None


def main():

    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

    # Params.
    p = get_arg_parser()
    opt = p.parse_args()
    opt.ibr_dataset = 1

    # Clear the stringified None.
    for k, v in vars(opt).items():
        if p.get_default(k) is None and v == 'None':
            setattr(opt, k, None)

    # Create log dir and copy the config file
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    print(f'Will log into {root_path}.')
    os.makedirs(root_path, exist_ok=True)
    f = os.path.join(root_path, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(opt)):
            attr = getattr(opt, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if opt.config_filepath is not None:
        f = os.path.join(root_path, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(opt.config_filepath, 'r').read())

    if opt.restart:
        print(f'Deleting previous logs in {root_path}...')
        common_utils.cond_rmtree(Path(root_path) / 'checkpoints')
        common_utils.cond_rmtree(Path(root_path) / 'summaries')
        common_utils.cond_rmtree(Path(root_path) / 'verbose')

    # Dataset.
    sdf_dataset = get_dataset(opt)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0) ##1 8

    # Model.
    model = get_sdf_decoder(opt, sdf_dataset)
    with (Path(root_path) / 'model.txt').open('w') as file:
        file.write(f'{model}')

    # Partial checkpoints.
    if opt.checkpoint_sdf and Path(opt.checkpoint_sdf).is_file():
        model.load_checkpoint(opt.checkpoint_sdf, load_sdf=True, load_poses=opt.load_model_poses)
    if opt.checkpoint_img_encoder and Path(opt.checkpoint_img_encoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_encoder, load_img_encoder=True, load_img_decoder=True)
    if opt.checkpoint_img_decoder and Path(opt.checkpoint_img_decoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_decoder, load_img_decoder=True)
    if opt.checkpoint_aggregation_mlp and Path(opt.checkpoint_aggregation_mlp).is_file():
        model.load_checkpoint(opt.checkpoint_aggregation_mlp, load_aggregation=True)

    # Resume?
    if opt.checkpoint_path:
        if not os.path.isfile(opt.checkpoint_path):
            raise RuntimeError(f"Could not find checkpoint {opt.checkpoint_path}.")
        checkpoint_file = Path(opt.checkpoint_path)
    else:
        checkpoint_file = get_latest_checkpoint_file(opt)
    if opt.resume and checkpoint_file and not (opt.fit_sphere or opt.fit_plane):
        print(f'Loading checkpoint from {checkpoint_file}...')
        model.load_checkpoint(checkpoint_file, strict=opt.checkpoint_strict)
    else:
        print('Starting training from scratch...')

    # Precompute 3D buffers if using pre-trained SDF network
    if opt.checkpoint_sdf and not opt.train_decoder_sdf:
        if opt.precomputed_3D_point_buffers and Path(opt.precomputed_3D_point_buffers).is_file() and opt.load_3D_point_buffers:
            print(f'Loading 3D position buffers from {opt.precomputed_3D_point_buffers}...')
            model.load_3D_buffers(opt.precomputed_3D_point_buffers)
        else:
            print(f'Precomputing 3D position buffers for this SDF model.')
            model.precompute_3D_buffers()
            if opt.save_3D_point_buffers:
                print(f'Saving 3D position buffers to {opt.precomputed_3D_point_buffers}.')
                model.save_3D_buffers(opt.precomputed_3D_point_buffers)

    # Define the loss
    loss_fn = partial(loss_functions.loss_sdf_ibr_mult, opt)
    summary_fn = partial(utils_ibr.write_sdf_color_summary_mult, opt, sdf_dataset)

    model.precompute_3D_buffers()

    # Define optimizer.
    if opt.feature_type == "rgb":
        opt.train_feature_decoder = 0
        opt.train_image_encoder = 0

    params = []
    if opt.train_decoder_sdf:
        params += [{'params': model.decoder_sdf.parameters(), 'name': 'sdf'}]
    if opt.train_feature_decoder:
        params += [{'params': model.dec_net.parameters(), 'name': 'image_dec'}]
    if opt.train_image_encoder:
        params += [{'params': model.enc_net.parameters(), 'name': 'image_enc'}]
    if opt.train_feature_blending and getattr(model, 'agg_net', False):
        params += [{'params': model.agg_net.parameters(), 'name': 'agg'}]
    if len(params) == 0:
        optimizer = None
        print(f'WARNING: Empty optimizer passed into training. No optimization will be performed.')
    else:
        optimizer = torch.optim.Adam(params=params, lr=opt.lr)

    # Train.
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                   clip_grad=True, optim=optimizer, verbose_record_file=opt.load_verbose_record,
                   ibr_log=True)

if __name__ == "__main__":
    main()
    print("OVERALL RUN TIME --- %s seconds ---" % (time.time() - start_time))