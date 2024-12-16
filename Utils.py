# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:06:41 2024
@author: zzha962
"""
import torchio as tio
from torchio.data.io import sitk_to_nib

def resample_nii(input_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None,
                 mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if (n != None):
        image = resampled_subject.img
        tensor_data = image.data
        if (isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[
            1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    return save_image


def data_preprocess(img_path):
    rm_IMG = resample_nii(img_path).as_sitk()
    sitk_image_arr, _ = sitk_to_nib(rm_IMG)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_image_arr))
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)
    
    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)
    roi_image = subject_roi.image.data.clone().detach().unsqueeze(1)
    return roi_image







