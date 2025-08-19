import h5py
import zarr
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nicety.conf import get_conf
from nicety.array import smallest_dtype


def convert_h5_to_zarr(h5_im_path, h5_mito_path, h5_mask_path, zarr_path):
    """Convert H5 files to zarr format with img and seg keys."""
    
    with h5py.File(h5_im_path, 'r') as f_im, h5py.File(h5_mito_path, 'r') as f_mito:
        # Get the main dataset from each file
        im_keys = list(f_im.keys())
        mito_keys = list(f_mito.keys())
        
        assert len(im_keys) == 1, f"Expected 1 key in {h5_im_path}, got {len(im_keys)}: {im_keys}"
        assert len(mito_keys) == 1, f"Expected 1 key in {h5_mito_path}, got {len(mito_keys)}: {mito_keys}"
        
        img_data = f_im[im_keys[0]][:]
        seg_data = f_mito[mito_keys[0]][:]
        if os.path.exists(h5_mask_path):
            print(f"Using mask file: {h5_mask_path}")
            with h5py.File(h5_mask_path, 'r') as f_mask:
                assert len(f_mask.keys()) == 1, f"Expected 1 key in {h5_mask_path}, got {len(f_mask.keys())}"
                mask_data = f_mask[list(f_mask.keys())[0]][:] > 0
        else:
            mask_data = np.ones_like(seg_data, dtype=bool)
        # mask = 0 implies seg = 0
        if not np.all((seg_data == 0) | mask_data):
            print("Warning: There are segmentation labels in regions where mask is 0. Setting those seg labels to -1.")

        if np.any(~mask_data):
            seg_data = seg_data.astype(smallest_dtype([seg_data, -1]))
            seg_data[~mask_data] = -1
        
        # Assert both original img and seg are 3D
        assert len(img_data.shape) == 3, f"Expected img to be 3D, got shape {img_data.shape}"
        assert len(seg_data.shape) == 3, f"Expected seg to be 3D, got shape {seg_data.shape}"
        
        # Add trailing dimension for images (should be 4D: x,y,z,channel)
        img_data = img_data[..., None]
        
        # Create zarr store
        root = zarr.open(zarr_path, mode='w')
        
        # Save img and seg
        root.create_array('img', data=img_data)
        root.create_array('seg', data=seg_data)


def create_skeleton_pkl(path):
    """Create empty skeleton.pkl file."""
    with open(path, 'wb') as f:
        pickle.dump({}, f)


def process_split(samples, data_path, split_dir, split_name):
    """Process a data split (train/val/test)."""
    os.makedirs(split_dir, exist_ok=True)
    
    for sample in tqdm(samples, desc=f"Processing {split_name}"):
        sample_dir = os.path.join(split_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Convert H5 to zarr
        h5_im_path = os.path.join(data_path, f'{sample}_im.h5')
        h5_mito_path = os.path.join(data_path, f'{sample}_mito.h5')
        h5_mask_path = os.path.join(data_path, f'{sample}_mask.h5')
        zarr_path = os.path.join(sample_dir, 'data.zarr')
        
        assert os.path.exists(h5_im_path), f"Image file not found: {h5_im_path}"
        assert os.path.exists(h5_mito_path), f"Mito file not found: {h5_mito_path}"
        
        convert_h5_to_zarr(h5_im_path, h5_mito_path, h5_mask_path, zarr_path)
        print(f"Converted {sample} to {zarr_path}")
        
        # Create skeleton.pkl
        skeleton_path = os.path.join(sample_dir, 'skeleton.pkl')
        create_skeleton_pkl(skeleton_path)


def process_mito_setting(setting_config, base_output_path):
    """Process a single mito setting configuration."""
    
    setting_name = setting_config.name
    data_path = setting_config.path
    output_dir = os.path.join(base_output_path, setting_name)
    
    # Process train/val/test splits
    process_split(setting_config.train, data_path, os.path.join(output_dir, 'train'), 'train')
    process_split(setting_config.val, data_path, os.path.join(output_dir, 'val'), 'val')
    process_split(setting_config.test, data_path, os.path.join(output_dir, 'test'), 'test')


if __name__ == "__main__":
    conf = get_conf()
    
    # Get mito configurations
    mito_configs = conf.mito.settings
    base_output_path = conf.mito.data_path
    
    # Process each mito setting
    for setting_config in tqdm(mito_configs, desc="Processing settings"):
        print(f"Processing setting: {setting_config.name}")
        process_mito_setting(setting_config, base_output_path)
    
    print("Data conversion completed!")
