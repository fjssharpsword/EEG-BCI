import numpy as np
import nibabel as nib
import json

def SALD_Case():

    #read MRI
    #https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html
    path = '/data/fjsdata/Brain-MRI/SALD/sub-031274_sub-031323/sub-031323/anat/sub-031323_T1w.nii.gz'
    img = nib.load(path)
    print(f'Type of the image {type(img)}')
    
    img_np = img.get_fdata()
    height, width, depth = img_np.shape
    print(f"The image object height: {height}, width:{width}, depth:{depth}")
    print(f'image value range: [{img_np.min()}, {img_np.max()}]')

    print(img.header.keys())
    pixdim =  img.header['pixdim']
    print(f'z轴分辨率： {pixdim[3]}')
    print(f'in plane 分辨率： {pixdim[1]} * {pixdim[2]}')
    z_range = pixdim[3] * depth
    x_range = pixdim[1] * height
    y_range = pixdim[2] * width
    print(x_range, y_range, z_range)

    #read json
    path = '/data/fjsdata/Brain-MRI/SALD/sub-031274_sub-031323/sub-031323/anat/sub-031323_T1w.json'
    with open(path, 'r') as jsn_file:
        jsn_data = json.load(jsn_file)
        print(jsn_data)

def main():
    SALD_Case()

if __name__ == "__main__":
    main()