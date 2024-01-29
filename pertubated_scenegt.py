# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates visibility, 2D bounding boxes etc. for the ground-truth poses.

See docs/bop_datasets_format.md for documentation of the calculated info.

The info is saved in folder "{train,val,test}_gt_info" in the main folder of the
selected dataset.
"""

import os
import numpy as np
import math

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

  # Whether to save visualizations of visibility masks.
  'vis_visibility_masks': False,

  # Tolerance used in the visibility test [mm].
  'delta': 15,

  # Type of the renderer.
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': '/media/DataSSD/TrainingData/bop_dataset',

  # Path template for output images with object masks.
  'vis_mask_visib_tpath': os.path.join(
    config.output_path, 'vis_gt_visib_delta={delta}',
    'vis_gt_visib_delta={delta}', '{dataset}', '{split}', '{scene_id:06d}',
    '{im_id:06d}_{gt_id:06d}.jpg'),
}
################################################################################

def perturb_rotation(axis, angle):
    """
    Generates a 3x3 rotation matrix for the given axis and angle.

    Args:
    axis (str): The rotation axis ('x', 'y', or 'z').
    angle (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")



def degrees_to_radians(degrees: float) -> float:
    """
    Converts an angle from degrees to radians.

    Args:
        degrees (float): The angle in degrees to convert.

    Returns:
        float: The equivalent angle in radians.
    """
    return degrees * math.pi / 180
    

def perturb_rt(rotation_matrix, translation_vector, angle_std_dev=0.01, trans_std_dev=0.1):
    """
    Perturbs a given rotation matrix and translation vector using a Gaussian distribution.

    Args:
    rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
    translation_vector (numpy.ndarray): A 3x1 translation vector.
    angle_std_dev (float, optional): Standard deviation for perturbing rotation angles.
                                   Defaults to 0.01 radians.
    trans_std_dev (float, optional): Standard deviation for perturbing translation vector elements.
                                   Defaults to 0.1.

    Returns:
    tuple: Perturbed rotation matrix and translation vector (perturbed_R, perturbed_T).
    """
    # Ensure the input is valid
    assert rotation_matrix.shape == (3, 3) and translation_vector.shape == (3, 1), "Invalid input dimensions."

    # Perturb rotation angles
    perturbed_angles = np.random.normal(0, angle_std_dev, 3)

    # Generate perturbed rotation matrices for each axis
    perturbed_rx = perturb_rotation('x', perturbed_angles[0])
    perturbed_ry = perturb_rotation('y', perturbed_angles[1])
    perturbed_rz = perturb_rotation('z', perturbed_angles[2])

    # Combine the perturbed rotation matrices
    perturbed_R = rotation_matrix @ perturbed_rx @ perturbed_ry @ perturbed_rz

    # Perturb translation vector
    perturbed_T = translation_vector + np.random.normal(0, trans_std_dev, (3, 1))

    return perturbed_R, perturbed_T


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

  scene_gt = inout.load_scene_gt(
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))

  scene_gt_pertubated = {}
  im_ids = sorted(scene_gt.keys())
  for im_counter, im_id in enumerate(im_ids):
    if im_counter % 100 == 0:
      misc.log(
        'Calculating GT info - dataset: {} ({}, {}), scene: {}, im: {}'.format(
          p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id,
          im_id))

    scene_gt_pertubated[im_id] = []
    for gt_id, gt in enumerate(scene_gt[im_id]):
        # in case of medical, we have one interation in this loop

        id = gt['obj_id']     
        R = gt['cam_R_m2c']
        T = gt['cam_t_m2c']

        R,T = perturb_rt(R, T, angle_std_dev=degrees_to_radians(2.5), trans_std_dev=8)


        scene_gt_pertubated[im_id].append({'obj_id': int(id), 
                                           'cam_R_m2c': [float(e) for e in R.reshape(9)], 
                                           'cam_t_m2c': [float(e) for e in T]})



  var = 25-0
  #Save the info for the current scene.
  scene_gt_pertubated_path = dp_split['scene_gt_tpath'].format(scene_id=scene_id)
  misc.ensure_dir(os.path.dirname(scene_gt_pertubated_path))
  inout.save_json(scene_gt_pertubated_path, scene_gt_pertubated)
