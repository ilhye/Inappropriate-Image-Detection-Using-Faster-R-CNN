import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_url, hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image


HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}

class RealESRGAN:
    def __init__(self, device, scale=2):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            hf_hub_download(config_file_url, cache_dir=cache_dir, force_filename=local_filename)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))
        
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    # @torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        torch.set_num_threads(8)

        scale = self.scale
        device = self.device

        print("Enter predict")
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        print("Split into patches")
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        print("Convert patches to tensor")
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device)

        print("Process in batches")
        with torch.no_grad(): 
            print("Processing patches")
            res = self.model(img[0:batch_size]) 
            print("Reshape")
            for i in range(batch_size, img.shape[0], batch_size): 
                print("Process patch")
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        print("Convert to image")
        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu().numpy()

        print("Stitch together")
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        sr_image = stich_together(
            sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )

        print("Unpad image")
        sr_img = unpad_image(sr_image, pad_size * scale)
        sr_img = (sr_img * 255).astype(np.uint8)
        sr_img = Image.fromarray(sr_img)

        print("Predict finished")
        return sr_img