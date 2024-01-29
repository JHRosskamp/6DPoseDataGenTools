# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates masks of object models in the ground-truth poses."""

import os
import numpy as np
import scipy.ndimage

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'medical',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'train',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': 'pbr',

  # Tolerance used in the visibility test [mm].
  'delta': 15,  # 5 for ITODD, 15 for the other datasets.

  # Type of the renderer.
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': '/media/data/TrainingData/bop_dataset',
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = None
if p['dataset'] == 'tless':
  model_type = 'cad'
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:

  # Load scene GT.
  scene_gt_path = dp_split['scene_gt_tpath'].format(
    scene_id=scene_id)
  scene_gt = inout.load_scene_gt(scene_gt_path)

  # Load scene camera.
  scene_camera_path = dp_split['scene_camera_tpath'].format(
    scene_id=scene_id)
  scene_camera = inout.load_scene_camera(scene_camera_path)

  # Create folders for the output masks (if they do not exist yet).
  mask_dir_path = os.path.dirname(
    dp_split['mask_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_dir_path)

  mask_visib_dir_path = os.path.dirname(
    dp_split['mask_visib_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_visib_dir_path)


  mask_complete_4_dir_path = os.path.dirname(
    dp_split['mask_complete_4_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_complete_4_dir_path)

  mask_complete_10_dir_path = os.path.dirname(
    dp_split['mask_complete_10_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_complete_10_dir_path)

  # Initialize a renderer.
  misc.log('Initializing renderer...')
  width, height = dp_split['im_size']
  ren = renderer.create_renderer(
    width, height, renderer_type=p['renderer_type'], mode='depth')

  # Add object models.
  #also markers as objects

  #list_object = [1,3,7,14,16]
  list_object = [1]
  for obj_id in list_object:#dp_model['obj_ids']:
    ren.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))
    for i in range(1,4):
        marker_id = i + obj_id * 1000
        print(marker_id)
        ren.add_object(marker_id, dp_model['model_tpath'].format(obj_id=marker_id))


  #sort file (should be sorted already)
  im_ids = sorted(scene_gt.keys())

  for im_id in im_ids:
  #for im_id in [im_ids[0], im_ids[1]]:


    if im_id % 100 == 0:
      misc.log(
        'Calculating masks - dataset: {} ({}, {}), scene: {}, im: {}'.format(
          p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id,
          im_id))

    K = scene_camera[im_id]['cam_K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Load depth image.
    depth_path = dp_split['depth_tpath'].format(
      scene_id=scene_id, im_id=im_id)
    depth_im = inout.load_depth(depth_path)
    depth_im *= scene_camera[im_id]['depth_scale']  # to [mm]
    dist_im = misc.depth_im_to_dist_im_fast(depth_im, K)


    # create new scene_gt with marker additions
    scene_gt_marker = scene_gt[im_id].copy()


    for ele in scene_gt[im_id]:
        id = ele['obj_id']
        t1 = np.array([0,0,0]).reshape(3,1) #marker and object are at the same position
        t0 = np.array(ele['cam_t_m2c']).reshape(3,1)
        for i in range(1,4):
           gt_marker = {}
           marker_id = i + id * 1000
           gt_marker['cam_R_m2c'] = ele['cam_R_m2c']
           gt_marker['cam_t_m2c'] = t0 + t1
           gt_marker['obj_id'] = marker_id
           scene_gt_marker.append(gt_marker)
        #print(scene_gt_marker)

 
    #print(scene_gt_marker)


    #for gt_id, gt in enumerate(scene_gt[im_id]):
    for gt_id, gt in enumerate(scene_gt_marker):

      # Render the depth image.
      depth_gt = ren.render_object(
        gt['obj_id'], gt['cam_R_m2c'], gt['cam_t_m2c'], fx, fy, cx, cy)['depth']

      # Convert depth image to distance image.
      dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)

      # Mask of the full object silhouette.
      mask = dist_gt > 0

      # Mask of the visible part of the object silhouette.
      mask_visib = visibility.estimate_visib_mask_gt(
        dist_im, dist_gt, p['delta'], visib_mode='bop19')

      # Save the calculated masks.
      mask_path = dp_split['mask_tpath'].format(
        scene_id=scene_id, im_id=im_id, gt_id=gt_id)
      inout.save_im(mask_path, 255 * mask.astype(np.uint8))

      mask_visib_path = dp_split['mask_visib_tpath'].format(
        scene_id=scene_id, im_id=im_id, gt_id=gt_id)
      inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))


    # open mask_visible for marker and calculate one final mask
    
    #load first mask of scene and image (always exists) and set to zero.
    mask_visib_path = dp_split['mask_visib_tpath'].format(
            scene_id=scene_id, im_id=im_id, gt_id=0)
    mask_visib_final = 0 * inout.load_im(mask_visib_path).astype(np.bool)

    for gt_id, gt in enumerate(scene_gt_marker):

        #marker has id > 1000
        if gt["obj_id"] > 1000:

            mask_visib_path = dp_split['mask_visib_tpath'].format(
            scene_id=scene_id, im_id=im_id, gt_id=gt_id)

            mask_visib = inout.load_im(mask_visib_path)
            mask_visib = mask_visib.astype(np.bool)

            #if i == 1:
            #    mask_visib_final = mask_visib
            #else:
            mask_visib_final = np.logical_or(mask_visib_final, mask_visib)

        
        # increase final mask by 2 pixels in each direction
        #mask_visib_final = scipy.ndimage.binary_dilation(mask_visib_final, iterations=2).astype(mask_visib_final.dtype)

    mask_visib_final_10 = mask_visib_final
    mask_visib_final_4 = mask_visib_final

    mask_visib_final_10 = scipy.ndimage.binary_dilation(mask_visib_final_10, iterations=9).astype(mask_visib_final.dtype)
    mask_visib_final_4 = scipy.ndimage.binary_dilation(mask_visib_final_4, iterations=4).astype(mask_visib_final.dtype)


    mask_visib_final_path = dp_split['mask_complete_4_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=0)
    inout.save_im(mask_visib_final_path, 255 * mask_visib_final_4.astype(np.uint8))

    mask_visib_final_path = dp_split['mask_complete_10_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=0)
    inout.save_im(mask_visib_final_path, 255 * mask_visib_final_10.astype(np.uint8))
