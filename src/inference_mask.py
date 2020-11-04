"""
	Modified from inference_dcm.py
	Run Inference 
"""
import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd
from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]
    hdr = dcmlist[0]
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    volume_ant = (pred==1).sum()
    volume_post = (pred==2).sum()
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def get_series_for_inference(path):
    dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    series_for_inference = [dcm for dcm in dicoms if dcm.SeriesDescription == 'HippoCrop']

    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def med_reshape(image, new_shape):

    reshaped_image = np.zeros(new_shape)

    x,y,z = image.shape
    reshaped_image[:x,:y,:z] = image

    return reshaped_image


def os_command(command):
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

if __name__ == "__main__":
    study_dir = '../data/TestVolumes/Study1/13_HCropVolume'
    # study_dir = "/data7/common/inqlee0704/HippoSeg/data/TestVolumes/Study1/13_HCropVolume" 

    print(f"Looking for series to run inference on in directory {study_dir}...")

    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=r"")
    # inference_agent = UNetInferenceAgent(
    #     device="cpu",
    #     parameter_file_path=r"/data7/common/inqlee0704/HippoSeg/train/RESULTS/2020-07-16_0717_Basic_unet/model.pth")

    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)

    # new_volume = med_reshape(volume,pred_label.shape)
    # print(new_volume.shape)
    # print(pred_label.shape)
    # img = nib.Nifti1Image(new_volume,affine=np.eye(4))
    # img.to_filename('YOUR PATH/volume.nii.gz')

