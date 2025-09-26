import numpy as np
from wmhparc.utils import save_manipulated_sitk_image_array, load_image
import SimpleITK as sitk
import pandas as pd

regions = {
    1 : "frontal-left",
    2 : "parietal-left",
    3 : "temporal-left",
    4 : "occipital-left",

    5 : "bgit",

    8 : "frontal-right",
    9 : "parietal-right",
    10 : "temporal-right",
    11 : "occipital-right",
}

rings = {
    1:'layer1',
    2:'layer2',
    3:'layer3',
    4:'layer4',
}

BRAIN_ROIS = {
    1: 'frontal-left_layer1',
    2: 'frontal-left_layer2',
    3: 'frontal-left_layer3',
    4: 'frontal-left_layer4',
    5: 'parietal-left_layer1',
    6: 'parietal-left_layer2',
    7: 'parietal-left_layer3',
    8: 'parietal-left_layer4',
    9: 'temporal-left_layer1',
    10: 'temporal-left_layer2',
    11: 'temporal-left_layer3',
    12: 'temporal-left_layer4',
    13: 'occipital-left_layer1',
    14: 'occipital-left_layer2',
    15: 'occipital-left_layer3',
    16: 'occipital-left_layer4',
    17: 'bgit_layer1',
    18: 'bgit_layer2',
    19: 'bgit_layer3',
    20: 'bgit_layer4',
    21: 'frontal-right_layer1',
    22: 'frontal-right_layer2',
    23: 'frontal-right_layer3',
    24: 'frontal-right_layer4',
    25: 'parietal-right_layer1',
    26: 'parietal-right_layer2',
    27: 'parietal-right_layer3',
    28: 'parietal-right_layer4',
    29: 'temporal-right_layer1',
    30: 'temporal-right_layer2',
    31: 'temporal-right_layer3',
    32: 'temporal-right_layer4',
    33: 'occipital-right_layer1',
    34: 'occipital-right_layer2',
    35: 'occipital-right_layer3',
    36: 'occipital-right_layer4'
}

synthseg_regions = {
    2:   'Left-Cerebral-White-Matter',
    3:   'Left-Cerebral-Cortex',
    4:   'Left-Lateral-Ventricle',
    5:   'Left-Inf-Lat-Vent',
    7:   'Left-Cerebellum-White-Matter',
    8:   'Left-Cerebellum-Cortex',
    10:  'Left-Thalamus-Proper',
    11:  'Left-Caudate',
    12:  'Left-Putamen',
    13:  'Left-Pallidum',
    14:  '3rd-Ventricle',
    15:  '4th-Ventricle',
    16:  'Brain-Stem',
    17:  'Left-Hippocampus',
    18:  'Left-Amygdala',
    26:  'Left-Accumbens-area',
    28:  'Left-VentralDC',
    
    41:  'Right-Cerebral-White-Matter',
    42:  'Right-Cerebral-Cortex',
    43:  'Right-Lateral-Ventricle',
    44:  'Right-Inf-Lat-Vent',
    46:  'Right-Cerebellum-White-Matter',
    47:  'Right-Cerebellum-Cortex',
    49:  'Right-Thalamus-Proper',
    50:  'Right-Caudate',
    51:  'Right-Putamen',
    52:  'Right-Pallidum',
    53:  'Right-Hippocampus',
    54:  'Right-Amygdala',
    58:  'Right-Accumbens-area',
    60:  'Right-VentralDC',
}


def create_combined_regions(atlas, pvrings):
    img = np.zeros(atlas.shape)
    counter = 1
    for region in regions.keys():
        for ring in rings.keys():
            img[(pvrings==ring) & (atlas==region)] = counter
            counter += 1

    return img

def save_brain_parcellation_image(atlas_path, pvrings_path):
    atlas_img = sitk.ReadImage(atlas_path)
    atlas = sitk.GetArrayFromImage(atlas_img)
    pvrings = load_image(pvrings_path)

    out_path = pvrings_path.split("pvrings")[0]  + "bullseye_parc.nii.gz"

    brain_rois = create_combined_regions(atlas, pvrings)

    save_manipulated_sitk_image_array(atlas_img, brain_rois, out_path)

    print("saved bullseye parcellation image to: ", out_path)
    return out_path

def parcellate_from_brainroi(brainroi, label, voxel_size, prefix="wmh"):
    results = {}
    for region_id, region_name in BRAIN_ROIS.items():
        results[f'{prefix}_{region_name}'] = np.sum(label[brainroi==region_id]) * voxel_size
    return results

def volumes_from_lobe_atlas(atlas, label, voxel_size, prefix='gray-m'):
    results = {}
    for region_id, region_name in regions.items():
        results[f'{prefix}_{region_name}'] = np.sum(label[atlas==region_id]) * voxel_size
    return results

def volumes_from_synthseg(synthseg, voxel_size):
    results = {}
    for region_id, region_name in synthseg_regions.items():
        results['synthseg_' + region_name] = np.sum(synthseg == region_id) * voxel_size
    return results

def get_ICV(brainmask, voxel_size):
    return {'icv': np.sum(brainmask != 0) * voxel_size}


def calc_parc_stats(image, parc_file, wmh_seg):
    brainroi = load_image(parc_file)
    wmh = load_image(wmh_seg)
    voxel_size = np.prod(sitk.ReadImage(image).GetSpacing())
    
    results = parcellate_from_brainroi(brainroi, wmh, voxel_size)
    results = {key:[value] for key, value in results.items()}
    return pd.DataFrame(results)

    return results


def get_all_brain_volumes(data):
    """
    combines the: WMH parcellation,
    gray matter cortex volumes per lobe and hemisphere
    cerebral white matter per lobe and hemisphere
    ICV
    synthseg_regions

    data: a dictionary that contains paths to the synthseg, brainroi, brainmask, brainatlas files and also voxel size
    
    """
    wmh_parc = parcellate_from_brainroi(data['brainroi'], data['wmh'], data['voxel_size'])
    wmh_parc['wmh_total'] = np.sum(data['wmh']) * data['voxel_size']
    gm = (data['synthseg'] == 3) | (data['synthseg'] == 42)
    wm = (data['synthseg'] == 2) | (data['synthseg'] == 41)
    gm_lobes = volumes_from_lobe_atlas(data['atlas'], gm, data['voxel_size'], prefix='gray-m-cerebral-cortex')
    wm_lobes = volumes_from_lobe_atlas(data['atlas'], wm, data['voxel_size'], prefix='white-m_cerebral')
    icv = get_ICV(data['brainmask'], data['voxel_size'])
    synthseg_regions = volumes_from_synthseg(data['synthseg'], data['voxel_size'])
    
    combined = (wmh_parc | gm_lobes | wm_lobes | icv | synthseg_regions)
    return combined
    # return wmh_parc, gm_lobes, wm_lobes, icv, synthseg_regions 

def parcellate_wmh(atlas, pvrings, wmh, voxel_size):
    results = {}
    for ring in rings.keys():
        for region in regions.keys():
            results[f'{regions[region]}_{rings[ring]}'] = np.sum(wmh[(pvrings==ring) & (atlas==region)]) * voxel_size

    return results
