""" Scan outlier metrics
"""


import numpy as np

import nibabel as nib

import nipraxis as npx


def dvars(img):
    """ Calculate dvars metric on Nibabel image `img`

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).

    Parameters
    ----------
    img : nibabel image

    Returns
    -------
    dvar_val : 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in `img`.
    """

    # read the image file
    img = img.get_fdata()

    # get the shape of the 3D volumes (e.g. without the time dimension)
    vol_shape = img.shape[:-1]

    # calculate the number of voxels
    n_voxels = np.prod(vol_shape)
    
    # create a voxel by time array (one 3D volume in each column)
    voxel_by_time = np.reshape(img, (n_voxels, img.shape[-1]))

    # 
    vol_diff = np.diff(voxel_by_time)
    dvar_val = np.sqrt(np.mean(vol_diff ** 2, axis = 0)) 

    return dvar_val 