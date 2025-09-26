"""
Register the template image to the target image.

Apply the registration transform to the atlas image.
"""
import os
from wmhparc.registration import run_ants_SyNAggro, apply_ants_transforms
from wmhparc.concentric_layers import postprocess_synthseg, create_pv_dist_ring_file
from wmhparc.parcellate_image import save_brain_parcellation_image, calc_parc_stats
import argparse

def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, type=str, help="path to the anatomical subject image of interest (e.g T1w or FLAIR image)")
    parser.add_argument('-b', '--brainmask', required=True, type=str, help="path to the brainmask (ICV) file of the subject image")
    parser.add_argument('-s', '--synthseg', required=True, type=str, help="path to the SynthSeg output of the subject anaotmical image")
    parser.add_argument('-t', '--template', required=True, type=str, help="path to the 73yr T1w template image")
    parser.add_argument('-a', '--atlas', required=True, type=str, help="path to the brainlobe atlas")
    parser.add_argument('-tb', '--template_brainmask', default=None, type=str, help="path to the brainmask (ICV) for the template image")
    parser.add_argument('-w', '--wmh_seg', required=True, type=str, help="path to the WMH segmentation file")
    parser.add_argument('-o', '--output_folder', required=True, type=str, help="output folder to save results to")

    return parser

def register_and_apply(image, template, atlas, output_folder, image_mask=None, template_mask=None):

    out_image = os.path.join(output_folder, image.split(os.path.sep)[-1].split(".nii")[0] + "_lobe_atlas.nii.gz")
    run_ants_SyNAggro(fixed=image, moving=template, out=out_image, outsuffix="template_synaggro", mask=image_mask, moving_mask=template_mask)

    affine_transform = out_image.split(".nii")[0] + "_template_synaggro_0GenericAffine.mat"
    warp_transform = out_image.split(".nii")[0] + "_template_synaggro_1Warp.nii.gz"

    print("applying ants transform")
    apply_ants_transforms(image, atlas, out_image, [affine_transform, warp_transform], is_label=True)

    print("transformed atlas saved to: ", out_image)
    return out_image 

def compute_concentric_layers(image, synthseg, brainmask, output_folder):
    print("computing ventricle and cortex distance transforms")
    ventmap_outimage, cortexmap_outimage = postprocess_synthseg(image, synthseg, output_folder)

    print("creating concentric layers images")
    pv_rings_file = create_pv_dist_ring_file(image, synthseg, ventmap_outimage, cortexmap_outimage, brainmask, output_folder)

    return pv_rings_file

def main(args):

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    
    # registration
    registered_atlas_file = register_and_apply(args.image, args.template, args.atlas, args.output_folder, image_mask=args.brainmask, template_mask=args.template_brainmask)

    # create concentric rings
    pv_rings_file = compute_concentric_layers(args.image, args.synthseg, args.brainmask, args.output_folder)

    # create bullseye parcellation image
    parc_file = save_brain_parcellation_image(registered_atlas_file, pv_rings_file)

    # calculate parcellation stats
    df = calc_parc_stats(args.image, parc_file, args.wmh_seg)

    df.to_csv(parc_file.split(".nii")[0] + "_wmh_vols.csv")

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
