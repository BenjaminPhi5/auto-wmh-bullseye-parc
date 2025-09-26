import SimpleITK as sitk
import os
import subprocess

def load_image(filepath):
    return sitk.GetArrayFromImage(sitk.ReadImage(filepath))

def save_manipulated_sitk_image_array(source_image, target_array, filepath):
    """Saves a manipulated nifti image array using the meta data information from a source image"""
    target_image = sitk.GetImageFromArray(target_array)
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetOrigin(source_image.GetOrigin())
    target_image.SetDirection(source_image.GetDirection())
    target_image.CopyInformation(source_image)
    sitk.WriteImage(target_image, filepath)

def resample_match_if_necessary(fixed, moving, use_nearest_neighbor=False):
    """
    resample moving to fixed if moving and fixed are not in the same space.
    """
    fixed_spacing = sitk.ReadImage(fixed).GetSpacing()
    moving_spacing = sitk.ReadImage(moving).GetSpacing()

    if (abs(fixed_spacing[0] - moving_spacing[0]) <= 0.05) and (abs(fixed_spacing[1] - moving_spacing[1]) <= 0.05) and (abs(fixed_spacing[2] - moving_spacing[2]) <= 0.05):
        return
    
    interp = "trilinear"
    if use_nearest_neighbor:
        interp = "nearest"

    command = [
        "mri_vol2vol", "--mov", moving, "--targ", fixed, "--o", moving, "--regheader", "--interp", interp
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)

def fileending(filepath):
    ending = ".nii" if filepath.endswith(".nii") else ".nii.gz" if filepath.endswith(".nii.gz") else None
    if ending is None:
        raise ValueError("file ending must be .nii or .nii.gz")
    return ending