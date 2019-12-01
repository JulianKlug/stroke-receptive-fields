#!/usr/bin/env python
import argparse, sys, os
import itk
import nibabel as nib
import numpy as np
from distutils.version import StrictVersion as VS
if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)


def segment_hessian_vessels(input_image, output_image, sigma=1.0, alpha1=0.5, alpha2=2.0):
    input_image = itk.imread(input_image, itk.ctype('float'))

    # apply hessian tube filter
    hessian_image = itk.hessian_recursive_gaussian_image_filter(input_image, sigma=sigma)

    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype('float')].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)

    itk.imwrite(vesselness_filter, output_image)

    # threshold the resulting image
    filtered_vessels_image = nib.load(output_image)
    filtered_vessels = filtered_vessels_image.get_data()
    segmented_vessels = np.zeros(filtered_vessels.shape)
    segmented_vessels[filtered_vessels > 5] = 1

    coordinate_space = filtered_vessels_image.affine
    segmented_img = nib.Nifti1Image(segmented_vessels, affine=coordinate_space)
    nib.save(segmented_img, output_image)

    return segmented_vessels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment blood vessels.')
    parser.add_argument('input_image')
    parser.add_argument('output_image')
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--alpha2', type=float, default=2.0)
    args = parser.parse_args()
    segment_hessian_vessels(args.input_image, args.output_image, args.sigma, args.alpha1, args.alpha2)


