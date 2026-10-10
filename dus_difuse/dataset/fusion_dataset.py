import ast
import os
import cv2
import math
import random
import numpy as np
import torch
import torch.utils.data as data


from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)
from .motionblur.motionblur import Kernel

from .degradation_sys import (
    add_noise_to_vi_img,
    add_blur_to_vi_img,
    add_rain_to_vi_img,
    add_snow_to_vi_img,
    add_haze_to_vi_img,
    add_noise_to_ir_img,
    add_stripe_to_ir_image,
    add_lowcontrast_to_ir_img
)

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

def _imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _resize_(imgs, crop_size=256):
    H, W = imgs[0].shape[:2]
    if H < crop_size or W < crop_size:
        scale = crop_size / min(H, W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        imgs = [
            cv2.resize(im, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
            for im in imgs
        ]
    return imgs


def _random_crop_(imgs, crop_size=256):
    H, W = imgs[0].shape[:2]
    if H == crop_size and W == crop_size:
        return imgs
    x = random.randint(0, W - crop_size)
    y = random.randint(0, H - crop_size)

    cropped_imgs = [im[y:y+crop_size, x:x+crop_size] for im in imgs]
    return cropped_imgs
    
def _clip_crop_and_resize_(imgs, min_crop_h=224, min_crop_w=224, clip_size=224):
    H, W = imgs[0].shape[:2]
    crop_h = random.randint(min_crop_h, H)
    crop_w = random.randint(min_crop_w, W)
    y = random.randint(0, H - crop_h)
    x = random.randint(0, W - crop_w)
    cropped_imgs = [im[y:y+crop_h, x:x+crop_w] for im in imgs]
    resized_imgs = [cv2.resize(im, (clip_size, clip_size), interpolation=cv2.INTER_CUBIC) for im in cropped_imgs]
    return resized_imgs

def _clip_resize_(imgs, clip_size=224):
    resized_imgs = [cv2.resize(im, (clip_size, clip_size), interpolation=cv2.INTER_CUBIC) for im in imgs]
    return resized_imgs

def _hflip(*imgs):
    return tuple(cv2.flip(im, 1) for im in imgs)

def _to_tensor_norm_neg1_1(img):
    # img: uint8 RGB, HxWx3
    arr = (img.astype(np.float32) / 255.0) * 2.0 - 1.0  # -> [-1,1]
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return torch.from_numpy(arr)

class FusionTrainingDataset(data.Dataset):
    def __init__(self, Dataset_path: str, crop_size: int = 256, hflip_prob: float = 0.2, return_params = False, om_degradation = False, 
        blur_kernel_size = 41,
        kernel_list = ['iso', 'aniso'],
        kernel_prob = [0.5, 0.5],
        blur_sigma = [0.1, 6],
        downsample_range = [0.8, 6], # 1,16
        noise_range = [0,6], # 0 15
        jpeg_range =  [60,100],
        motion_kernel_path = None,
        motion_prob = [0.25,0.005],
        motion_kernel_range = [1,9],
        motion_intensity_range = [0,1],
        visible_input_dir = None,
        visible_target_dir = None,
        visible_metadata = None,
    ):
        super().__init__()
        self.root = Dataset_path
        self.dir_depth = os.path.join(Dataset_path, "depth")
        self.dir_vi_lq = self._resolve_data_path(
            Dataset_path, visible_input_dir, ("vis", "vi_lq"), "visible input folder"
        )
        self.dir_vi_hq = self._resolve_data_path(
            Dataset_path, visible_target_dir, ("vis_enhance", "vi_hq"),
            "visible target folder",
        )
        self.dir_ir    = os.path.join(Dataset_path, "ir")
        self.return_params = return_params

        self.crop_size = crop_size
        self.hflip_prob = hflip_prob

        if not os.path.isdir(self.dir_ir):
            raise FileNotFoundError(f"IR folder not found: {self.dir_ir}")

        files = [f for f in os.listdir(self.dir_ir) if os.path.splitext(f)[-1].lower() in IMG_EXTS]
        files.sort()
        self.data = files
        
        
        self.ir_params = {}
        self.vi_lq_params = {}
        if self.return_params:
            ir_txt_path = os.path.join(Dataset_path, "ir.txt")
            vi_txt_path = self._resolve_data_path(
                Dataset_path, visible_metadata, ("vis.txt", "vi_lq.txt"),
                "visible metadata file", required=False,
            )
            self.ir_params = self._read_params_txt(ir_txt_path)
            self.vi_lq_params = self._read_params_txt(vi_txt_path)
            
        self.om_degradation = om_degradation
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.motion_prob = motion_prob
        self.motion_kernel_range = motion_kernel_range
        self.motion_intensity_range = motion_intensity_range
        self.motion_kernels = (
            torch.load(motion_kernel_path, map_location="cpu", weights_only=False)
            if motion_kernel_path and os.path.isfile(motion_kernel_path)
            else None
        )

    @staticmethod
    def _resolve_data_path(root, configured_name, candidates, description, required=True):
        """Resolve an explicit entry or auto-detect the current and legacy names."""
        if configured_name:
            path = configured_name
            if not os.path.isabs(path):
                path = os.path.join(root, path)
            if required and not os.path.exists(path):
                raise FileNotFoundError(f"{description.capitalize()} not found: {path}")
            return path
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
        path = os.path.join(root, candidates[0])
        if required:
            choices = ", ".join(candidates)
            raise FileNotFoundError(
                f"Could not find {description} under {root}; expected one of: {choices}"
            )
        return path
        
        

    def apply_degradation(self, img_gt):
        """
        Args:s
            img_gt: numpy array (H, W, C) in RGB format, range [0, 1]
        Returns:
            img_lq: degraded image, same format as input
        """
        h, w, _ = img_gt.shape
        
        r = random.random()
        if r < 0.3:
            level = 'light'
        elif r < 0.6:
            level = 'heavy'
        else:
            return img_gt
    
        if level == 'light':
            blur_sigma_get = [self.blur_sigma[0], self.blur_sigma[1] / 2]
            downsample_range_get = [self.downsample_range[0], self.downsample_range[1] / 2]
            noise_range_get = [self.noise_range[0], self.noise_range[1] / 2]
            jpeg_range_get = [min(self.jpeg_range[0] * 1.2, 100), self.jpeg_range[1]]
    
        elif level == 'heavy':  # heavy
            blur_sigma_get = [self.blur_sigma[0], self.blur_sigma[1]]
            downsample_range_get = [self.downsample_range[0], self.downsample_range[1]]
            noise_range_get = [self.noise_range[0], self.noise_range[1]]
            jpeg_range_get = [self.jpeg_range[0], self.jpeg_range[1]]

        
        #motion blur
        if np.random.rand() < self.motion_prob[0]:
            if self.motion_kernels is not None and np.random.rand() < self.motion_prob[1]:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_lq = cv2.filter2D(img_gt, -1, k)
            else:
                size_sample = np.random.randint(self.motion_kernel_range[0], self.motion_kernel_range[1]+1)
                intensity_sample = np.random.uniform(self.motion_intensity_range[0], self.motion_intensity_range[1])
                k = Kernel(size=(size_sample, size_sample), intensity=intensity_sample).kernelMatrix
                img_lq = cv2.filter2D(img_gt, -1, k)
        else:
            img_lq = img_gt
        

        
        #blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            blur_sigma_get,
            blur_sigma_get,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_lq, -1, kernel)
        
        # downsample
        scale = np.random.uniform(downsample_range_get[0], downsample_range_get[1])
        img_lq = cv2.resize(
            img_lq, 
            (int(w // scale), int(h // scale)), 
            interpolation=cv2.INTER_LINEAR
        )
        
        #noise
        if noise_range_get is not None:
            img_lq = random_add_gaussian_noise(img_lq, noise_range_get)
        
        #jpeg compression
        if jpeg_range_get is not None:
            img_lq = random_add_jpg_compression(img_lq, jpeg_range_get)
        
        #resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return img_lq


    def _read_params_txt(self, path):
        params_dict = {}
        if not os.path.isfile(path):
            return params_dict
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                info = ast.literal_eval(line)
                params_dict[info["name"]] = {"type": info["type"], "params": info["params"]}
        return params_dict

    def __len__(self):
        return len(self.data)

    def apply_vi_degrades(self, vi_img, depth_3ch, return_params=False, vi_lq_params = None):
        ops = [
            ("noise", 0, lambda img: add_noise_to_vi_img(img, return_params=return_params)),
            ("blur" , 1, lambda img: add_blur_to_vi_img(img, return_params=return_params)),
            ("rain" , 2, lambda img: add_rain_to_vi_img(img, return_params=return_params)),
            ("snow" , 3, lambda img: add_snow_to_vi_img(img, return_params=return_params)),
        ]

        if depth_3ch is not None:
            ops.append(("haze", 4, lambda img: add_haze_to_vi_img(img, depth_3ch, return_params=return_params)))
        name, type_id, fn = random.choice(ops)
        if return_params:
            out, degra_score = fn(vi_img)
            vi_lq_params["type"] = vi_lq_params["type"] * 5 + type_id
            vi_lq_params["params"] = vi_lq_params["params"] + degra_score
        else:
            out = fn(vi_img)
        return out, vi_lq_params

    def apply_ir_degrades(self, ir_img, return_params=False, ir_params = None):
        ops = [
            ("noise", 0, lambda img: add_noise_to_ir_img(img, return_params=return_params)),
            ("lowcontrast", 1, lambda img: add_lowcontrast_to_ir_img(img, return_params=return_params)),
            ("stripe", 2, lambda img: add_stripe_to_ir_image(img, return_params=return_params)),
        ]
        name, type_id, fn = random.choice(ops)
        if return_params:
            out, degra_score = fn(ir_img)
            ir_params["type"] = ( ir_params["type"] + 1) * 10 + type_id
            ir_params["params"] = ir_params["params"] + degra_score
        else:
            out = fn(ir_img)
        
        return out, ir_params
    # --------------------------------------------------------------------

    def __getitem__(self, index: int):
        fname = self.data[index]
        p_ir    = os.path.join(self.dir_ir, fname)
        p_depth = os.path.join(self.dir_depth, fname)
        p_vi_lq = os.path.join(self.dir_vi_lq, fname)
        p_vi_hq = os.path.join(self.dir_vi_hq, fname)

        # read
        img_ir_hq = _imread_color(p_ir)            # IR
        img_vi_hq = _imread_color(p_vi_hq)         # GT VIS
        img_vi_lq_in = _imread_color(p_vi_lq)      # LQ VIS
        depth = _imread_color(p_depth)              # depth
        
        ir_params = dict(self.ir_params.get(fname, {"type": 0, "params": 0.0}))
        vi_lq_params = dict(self.vi_lq_params.get(fname, {"type": 0, "params": 0.0}))
        
        ir_hq_resized, vi_hq_resized, vi_lq_in_resized, depth_resized = _resize_([img_ir_hq, img_vi_hq, img_vi_lq_in, depth], crop_size=self.crop_size)
        ir_lq_in_resized = ir_hq_resized.copy()
        vi_lq_in_resized_copy = vi_lq_in_resized.copy()
        ir_lq_in_resized_copy = ir_hq_resized.copy()

        vi_ir_tran = (random.random() < 0.5)
        if vi_ir_tran:
            new_vi_hq_resized = ir_hq_resized
            new_vi_lq_in_resized = ir_lq_in_resized
            new_vi_lq_in_resized_copy = ir_lq_in_resized_copy
            new_vi_lq_params = dict(ir_params)
        else:
            new_vi_hq_resized = vi_hq_resized
            new_vi_lq_in_resized = vi_lq_in_resized
            new_vi_lq_in_resized_copy = vi_lq_in_resized_copy   
            new_vi_lq_params = dict(vi_lq_params)
        ir_vi_tran = (random.random() < 0.5)

        if ir_vi_tran:
            new_ir_hq_resized = vi_hq_resized
            new_ir_lq_in_resized = vi_lq_in_resized
            new_ir_lq_in_resized_copy = vi_lq_in_resized_copy
            new_ir_params = dict(vi_lq_params)
        else:
            new_ir_hq_resized = ir_hq_resized
            new_ir_lq_in_resized = ir_lq_in_resized
            new_ir_lq_in_resized_copy = ir_lq_in_resized_copy       
            new_ir_params = dict(ir_params)

        
        if self.om_degradation:
            new_vi_lq_in_resized = self.apply_degradation((new_vi_lq_in_resized / 255.0).astype(np.float32))
            new_ir_lq_in_resized = self.apply_degradation((new_ir_lq_in_resized / 255.0).astype(np.float32))
            
            new_vi_lq_in_resized = (new_vi_lq_in_resized.clip(0, 1) * 255).astype(np.float32)
            new_ir_lq_in_resized = (new_ir_lq_in_resized.clip(0, 1) * 255).astype(np.float32)
            
        else:
            new_vi_lq_in_resized = new_vi_lq_in_resized
            new_ir_lq_in_resized = new_ir_lq_in_resized

        if vi_ir_tran:
            new_vi_lq_resized, new_vi_lq_params = self.apply_ir_degrades(new_vi_lq_in_resized.copy(), self.return_params, new_vi_lq_params)
        else:
            new_vi_lq_resized, new_vi_lq_params = self.apply_vi_degrades(new_vi_lq_in_resized.copy(), depth_resized, self.return_params, new_vi_lq_params)
         
        if ir_vi_tran:
            new_ir_lq_resized, new_ir_lq_params = self.apply_vi_degrades(new_ir_lq_in_resized.copy(),depth_resized, self.return_params, new_ir_params)
        else:
            new_ir_lq_resized, new_ir_lq_params = self.apply_ir_degrades(new_ir_lq_in_resized.copy(), self.return_params, new_ir_params)
        
        # flip
        if random.random() < self.hflip_prob:
            new_vi_lq_in_resized_copy,new_ir_lq_in_resized_copy, new_vi_hq_resized, new_vi_lq_resized, new_ir_hq_resized, new_ir_lq_resized = _hflip(
                new_vi_lq_in_resized_copy,new_ir_lq_in_resized_copy, new_vi_hq_resized, new_vi_lq_resized, new_ir_hq_resized, new_ir_lq_resized
            )
        new_vi_lq_clip, new_ir_lq_clip = _clip_resize_(
            [new_vi_lq_resized, new_ir_lq_resized]
        )
        t_vi_lq_clip = _to_tensor_norm_neg1_1(new_vi_lq_clip)
        t_ir_lq_clip = _to_tensor_norm_neg1_1(new_ir_lq_clip)

        t_vi_lq_in = _to_tensor_norm_neg1_1(new_vi_lq_in_resized_copy)
        t_ir_lq_in = _to_tensor_norm_neg1_1(new_ir_lq_in_resized_copy)
        t_vi_hq = _to_tensor_norm_neg1_1(new_vi_hq_resized)
        t_vi_lq = _to_tensor_norm_neg1_1(new_vi_lq_resized)
        t_ir_hq = _to_tensor_norm_neg1_1(new_ir_hq_resized)
        t_ir_lq = _to_tensor_norm_neg1_1(new_ir_lq_resized)
        
        
        if self.return_params:
            return {
                "vi_lq_in":t_vi_lq_in,
                "ir_lq_in":t_ir_lq_in,
                "vi_hq": t_vi_hq,        # 3xHxW
                "vi_lq": t_vi_lq,        # 3xHxW
                "ir_hq": t_ir_hq,        # 3xHxW
                "ir_lq": t_ir_lq,        # 3xHxW
                "path": {
                    "ir": p_ir,
                    "depth": p_depth,
                    "vi_lq": p_vi_lq,
                    "vi_hq": p_vi_hq
                },
                "prompt": "",
                "vi_lq_params": new_vi_lq_params,
                "ir_lq_params": new_ir_lq_params,
                "vi_lq_clip": t_vi_lq_clip,
                "ir_lq_clip": t_ir_lq_clip,
            }
        else:
            return {
                "vi_hq": t_vi_hq,        # 3xHxW
                "vi_lq": t_vi_lq,        # 3xHxW
                "ir_hq": t_ir_hq,        # 3xHxW
                "ir_lq": t_ir_lq,        # 3xHxW
                "path": {
                    "ir": p_ir,
                    "depth": p_depth,
                    "vi_lq": p_vi_lq,
                    "vi_hq": p_vi_hq
                },
                "prompt": "",
                "vi_lq_clip": t_vi_lq_clip,
                "ir_lq_clip": t_ir_lq_clip,
            }
