import subprocess
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import os
from wmhparc.utils import resample_match_if_necessary, fileending, load_image, save_manipulated_sitk_image_array
import numpy as np


# values of ventricle and cortex values in the synthseg output
VENTRICLE_1 = 4
VENTRICLE_2 = 43
VENTRICLE_INFERIOR_L = 5
VENTRICLE_INFERIOR_R = 44

CORTEX_1 = 3
CORTEX_2 = 42
CORTEX_3 = 8
CORTEX_4 = 47
CORTEX_PARC = 1000


def create_ventricle_distance_map(synthseg_file, outfile_ventricle, outfile_cortex):
    """
    loads the synth_seg segmentation, extracts the ventricles segmentation and the cortex segmentation
    and creates a euclidian distance map from each voxel to the ventricles.
    This distance map is then saved under the name out_file
    """
    
    synthseg_img = sitk.ReadImage(synthseg_file)
    
    spacing = synthseg_img.GetSpacing()
    if not ((0.95 <= spacing[0] <= 1.05) and (0.95 <= spacing[1] <= 1.05) and (0.95 <= spacing[2] <= 1.05)):
        raise ValueError(f"image spacing must be approx (1, 1, 1) to compute distance map, not {spacing}")
        
    def extract_distance(condition, outfile):
        condition = condition.astype(np.float32)
        distance_map = distance_transform_edt(1 - condition)
        save_manipulated_sitk_image_array(synthseg_img, distance_map, outfile)
    
    synthseg = sitk.GetArrayFromImage(synthseg_img)
    vent_dist = extract_distance((synthseg == VENTRICLE_1) | (synthseg == VENTRICLE_2) | (synthseg == VENTRICLE_INFERIOR_L) | (synthseg == VENTRICLE_INFERIOR_R), outfile_ventricle)
    extract_distance((synthseg == CORTEX_1) | (synthseg == CORTEX_2) | (synthseg == CORTEX_3) | (synthseg == CORTEX_4) | (synthseg > CORTEX_PARC), outfile_cortex)

def postprocess_synthseg(in_image, synthseg_outimage, out_folder):
    """
    takes the synthseg output, creates the ventricle distance and cortex distance map
    and then resamples all three images to the space of the original input image synthseg was run on.

    in_image: the image that synthseg was run on
    out_folder: the path to the derivatives folder where the synthseg imgage is stored and the distance maps will be created.
    """
    in_imagename = in_image.split(".nii")[0].split("/")[-1]
    in_filetype = fileending(in_image)
    ventmap_outimage = os.path.join(out_folder, in_imagename + "_ventdist" + in_filetype)
    cortexmap_outimage = os.path.join(out_folder, in_imagename + "_cortexdist" + in_filetype)

    # ensure the synthseg image is in 1x1x1 space
    command = [
        "mri_convert", "-vs", "1", "1", "1", "-rt", "nearest",
        synthseg_outimage, ventmap_outimage
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    # create ventricle distance map imag
    create_ventricle_distance_map(ventmap_outimage, ventmap_outimage, cortexmap_outimage)

    # resample the output images back to the space of the in_image
    resample_match_if_necessary(in_image, ventmap_outimage, use_nearest_neighbor=False)
    resample_match_if_necessary(in_image, cortexmap_outimage, use_nearest_neighbor=False)

    return ventmap_outimage, cortexmap_outimage

"""
calculates the ventricle rings (rings of distance from the ventricles for a brain mri image (preferably a t1 image)

includes function to load the distance images from disk, calculate the rings and save result to disk.
"""

def combined_roi_array(lobes):
    arr = np.zeros(lobes[0].shape)
    
    for idx, lobe in enumerate(lobes):
        arr[lobe] = idx + 1
        
    return arr

def compute_pv_distance_rings(vent_dist, cortex_dist, brainmask):
    norm_dist = vent_dist / (vent_dist + cortex_dist)
    ring1 = (norm_dist < 0.25) & brainmask
    ring2 = (norm_dist >= 0.25) & (norm_dist < 0.5) & brainmask
    ring3 = (norm_dist >= 0.5) & (norm_dist < 0.75) & brainmask
    ring4 = (norm_dist >= 0.75) & brainmask
    
    rings = combined_roi_array([ring1, ring2, ring3, ring4])
    
    return rings, norm_dist

def create_pv_dist_ring_file(in_image, synthseg_outimage, ventmap_outimage, cortexmap_outimage, brainmask_outimage, out_folder):
    """
    takes the vent and cortex dist maps, creates the pv ring maps
    save to disk as a file name pvrings, copying the metadata from the original synthseg image

    in_image: the image that synthseg was run on
    out_folder: the path to the derivatives folder where the synthseg imgage is stored and the distance maps will be created.
    """
    in_imagename = in_image.split(".nii")[0].split("/")[-1]
    in_filetype = fileending(in_image)

    pvrings_outimage = os.path.join(out_folder, in_imagename + "_pvrings" + in_filetype)
    
    synthseg_itk_img = sitk.ReadImage(synthseg_outimage)
    vent_dist = load_image(ventmap_outimage)
    cortex_dist = load_image(cortexmap_outimage)
    brainmask = load_image(brainmask_outimage) == 1
    
    pv_distance_rings, _ = compute_pv_distance_rings(vent_dist, cortex_dist, brainmask)
    
    save_manipulated_sitk_image_array(synthseg_itk_img, pv_distance_rings, pvrings_outimage)

    print("saved concentric layers segmentation to: ", pvrings_outimage)

    return pvrings_outimage


