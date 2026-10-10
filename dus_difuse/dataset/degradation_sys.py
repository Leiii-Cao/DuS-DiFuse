import cv2
import numpy as np
import random

#-----------------vis_noise------------------------------
def add_gaussian_noise(image, mean=0, std=20):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_poisson_noise(image, value=None):
    """Add Poisson noise to an image."""
    if value is None:
        value = np.random.randint(70, 91)
    noisy_image = np.random.poisson(image / 255.0 * value) / value * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_noise_to_vi_img(
    img,
    noise_types=("gaussian", "poisson"),
    gaussian_std_range=(0, 20),
    poisson_value_range=(60, 200),
    gaussian_probability=0.6,
    poisson_probability=0.6,
    return_params=False,
):
    img = img.astype(np.float32)
    noisy = img.copy()
    std = 0
    value = 0
    # --- Gaussian noise ---
    if "gaussian" in noise_types and random.random() < gaussian_probability:
        std = random.uniform(*gaussian_std_range)
        noisy = add_gaussian_noise(noisy, mean=0, std=std)

    # --- Poisson noise ---
    if "poisson" in noise_types and random.random() < poisson_probability:
        value = random.randint(*poisson_value_range)
        noisy = add_poisson_noise(noisy, value=value)

    if return_params:
        max_noisy = img.copy().astype(np.float32)
        if "gaussian" in noise_types:
            max_std = gaussian_std_range[1]
            max_noisy = add_gaussian_noise(max_noisy, mean=0, std=max_std).astype(np.float32)
        if "poisson" in noise_types:
            max_value = poisson_value_range[0]
            max_noisy = add_poisson_noise(max_noisy, value=max_value).astype(np.float32)
            
        l1_random = np.mean(np.abs(noisy.astype(np.float32) - img))
        l1_max = np.mean(np.abs(max_noisy.astype(np.float32) - img))
        deg_score = l1_random / (l1_max + 1e-8)
        deg_score = np.clip(deg_score, 0.0, 1.0)
        return noisy, deg_score


    return noisy


#----------------vis_blur----------------------------------

def motion_blur_kernel(length, angle):
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0  
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= kernel.sum()
    return kernel

def add_blur_to_vi_img(
    img, 
    motion_blur_prob=0.6, motion_blur_len=(1, 10), motion_blur_angle=(0, 180),
    gaussian_blur_prob=0.6, gaussian_blur_ksize=(3, 11),
    return_params=False
):
    img = img.astype(np.float32)
    out = img.copy()
    length = 0
    angle = 0
    ksize = 0
    if random.random() < motion_blur_prob:
        length = random.randint(*motion_blur_len)
        angle = random.uniform(*motion_blur_angle)
        
        kernel = motion_blur_kernel(length, angle)
        out = cv2.filter2D(out, -1, kernel)

    if random.random() < gaussian_blur_prob:
        ksize = random.randint(*gaussian_blur_ksize)
        
        if ksize % 2 == 0:  
            ksize += 1
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        out = cv2.GaussianBlur(out, (ksize, ksize), sigma)


    if return_params:
        max_out = img.copy().astype(np.float32)

        max_len = motion_blur_len[1]
        max_angle = random.uniform(*motion_blur_angle)
        k_max = motion_blur_kernel(max_len, max_angle)
        max_out = cv2.filter2D(max_out, -1, k_max)

        max_ksize = gaussian_blur_ksize[1]
        if max_ksize % 2 == 0:
            max_ksize += 1
        sigma_max = 0.3 * ((max_ksize - 1) * 0.5 - 1) + 0.8
        max_out = cv2.GaussianBlur(max_out, (max_ksize, max_ksize), sigma_max)

        l1_random = np.mean(np.abs(out.astype(np.float32) - img))
        l1_max = np.mean(np.abs(max_out.astype(np.float32) - img))

        deg_score = l1_random / (l1_max + 1e-8)
        deg_score = float(np.clip(deg_score, 0.0, 1.0))
        
        return out, deg_score


    return out
    


#----------------vis_rain----------------------------------

def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise
    

def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  
    k = cv2.warpAffine(dig, trans, (length, length))  
    k = cv2.GaussianBlur(k, (w, w), 0)  

    # k = k / length                       

    blurred = cv2.filter2D(noise, -1, k)  

 
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
    
def add_rain_to_vi_img(img,alpha_range=(0.1, 1.0),value_range=(100, 500),length_range=(30, 50),angle_range=(-30, 30),w_choices=(1, 3, 5),return_params=False):
    
    alpha  = round(random.uniform(*alpha_range), 2)
    value  = random.randint(*value_range)
    w      = random.choice(w_choices)
    if length_range is not None:
        length = random.randint(*length_range)
    else:
        length= w
    angle  = random.uniform(*angle_range)

    
    noise = get_noise(img, value)
    rain = rain_blur(noise, length, angle, w)

   
    rain_rgb = np.dstack([rain, rain, rain]).astype(np.float32) / 255.0
    m = np.where(rain_rgb > 0, 1, 0)
    img_f = img.astype(np.float32) / 255.0
    local = rain_rgb * m + (1 - rain_rgb) * img_f
    result = (1 - alpha) * img_f + alpha * local
    out = np.clip(result * 255, 0, 255).astype(np.uint8)
    if return_params:
        alpha_max  = alpha_range[1]
        value_max  = value_range[1]
        length_max = length_range[1] if length_range is not None else max(w_choices)
        w_max      = max(w_choices)

        noise_max = get_noise(img, value_max)
        rain_max  = rain_blur(noise_max, length_max, angle, w_max)
        rain_max_rgb = np.dstack([rain_max, rain_max, rain_max]).astype(np.float32) / 255.0
        m_max = np.where(rain_max_rgb > 0, 1, 0)
        local_max = rain_max_rgb * m_max + (1 - rain_max_rgb) * img_f
        result_max = (1 - alpha_max) * img_f + alpha_max * local_max
        out_max = np.clip(result_max * 255, 0, 255).astype(np.uint8)

        # L1-distance
        l1_random = np.mean(np.abs(out.astype(np.float32) - img.astype(np.float32)))
        l1_max    = np.mean(np.abs(out_max.astype(np.float32) - img.astype(np.float32)))

        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
        
        return out, deg_score


    return out
    
    
#----------------vis_snow----------------------------------

def add_snow_to_vi_img(img,alpha_range=(0.2, 1.0),value_range=(50, 100),angle_range=(-30, 30),w_choices=(25,27,29,31,33),return_params=False):
    
    alpha  = round(random.uniform(*alpha_range), 2)
    value  = random.randint(*value_range)
    w      = random.choice(w_choices)
    length= w
    angle  = random.uniform(*angle_range)

    

    noise = get_noise(img, value)
    rain = rain_blur(noise, length, angle, w)

   
    rain_rgb = np.dstack([rain, rain, rain]).astype(np.float32) / 255.0
    m = np.where(rain_rgb > 0, 1, 0)
    img_f = img.astype(np.float32) / 255.0
    local = rain_rgb * m + (1 - rain_rgb) * img_f
    result = (1 - alpha) * img_f + alpha * local
    out = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    if return_params:
        alpha_max  = alpha_range[1]
        value_max  = value_range[1]
        w_max      = max(w_choices)
        length_max = w_max

        noise_max = get_noise(img, value_max)
        snow_max  = rain_blur(noise_max, length_max, angle, w_max)
        snow_max_rgb = np.dstack([snow_max, snow_max, snow_max]).astype(np.float32) / 255.0
        m_max = np.where(snow_max_rgb > 0, 1, 0)
        local_max = snow_max_rgb * m_max + (1 - snow_max_rgb) * img_f
        result_max = (1 - alpha_max) * img_f + alpha_max * local_max
        out_max = np.clip(result_max * 255, 0, 255).astype(np.uint8)

        # L1-distance
        l1_random = np.mean(np.abs(out.astype(np.float32) - img.astype(np.float32)))
        l1_max    = np.mean(np.abs(out_max.astype(np.float32) - img.astype(np.float32)))

        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
        
        return out, deg_score

    return out
    
    
    
#----------------vis_haze----------------------------------

def add_haze_to_vi_img(img, depth,
        beta_max=2.5, beta_min=0.5,
        A_max=0.9, A_min=0.6,
        color_max=0.01, color_min=-0.01,return_params=False):
    """
    Apply synthetic haze to a single image using its depth map.

    Args:
        img (np.array): Input image (RGB, uint8, [0,255]).
        depth (np.array): Corresponding depth map (uint8, [0,255]).
        beta_max (float): Maximum scattering coefficient.
        beta_min (float): Minimum scattering coefficient.
        A_max (float): Maximum atmospheric light.
        A_min (float): Minimum atmospheric light.
        color_max (float): Maximum color distortion for atmospheric light.
        color_min (float): Minimum color distortion for atmospheric light.

    Returns:
        np.array: Hazy image (uint8, [0,255]).
    """
    # Normalize inputs to [0,1]
    img_norm = img.astype(np.float32) / 255.0
    depth_norm = depth.astype(np.float32) / 255.0
    # Random scattering coefficient
    beta = np.random.uniform(beta_min, beta_max)

    # Compute transmission map with blur
    t = np.exp(-np.minimum(1 - cv2.blur(depth_norm, (22, 22)), 0.7) * beta)

    # Random atmospheric light
    A_base = np.random.uniform(A_min, A_max)
    A_random_color = np.random.uniform(color_min, color_max, 3)
    A = A_base + A_random_color

    # Apply haze formula
    hazy = img_norm * t + A * (1 - t)

    # Convert back to uint8
    out = (hazy * 255.0).clip(0, 255).astype(np.uint8)
    
    
    if return_params:
       
        beta_max_val = beta_max
        A_max_val_base = A_max
        A_max_val_color = color_max

        t_max = np.exp(-np.minimum(1 - cv2.blur(depth_norm, (22, 22)), 0.7) * beta_max_val)
        A_max_val = A_max_val_base + np.array([A_max_val_color]*3)
        hazy_max = img_norm * t_max + A_max_val * (1 - t_max)
        out_max = (hazy_max * 255.0).clip(0, 255).astype(np.uint8)

        # L1-distance
        l1_random = np.mean(np.abs(out.astype(np.float32) - img.astype(np.float32)))
        l1_max = np.mean(np.abs(out_max.astype(np.float32) - img.astype(np.float32)))
        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
        
        return out, deg_score
        
    return out
    
    
    
    
#----------------ir_noise----------------------------------

def add_noise_to_ir_img(
        img,
        gaussian_mean_range=(0, 0),
        gaussian_std_range=(0, 20),
        gaussian_prob=0.6,
        poisson_min=60,
        poisson_max=200,
        poisson_prob=0.6,
        salt_prob_range=(0.0001, 0.005),
        pepper_prob_range=(0.0001, 0.005),
        sp_prob=0.6,
        return_params=False
):

    img = img.astype(np.float32)
    H, W, _ = img.shape

  
    gaussian_mean = np.random.uniform(*gaussian_mean_range)
    gaussian_std  = np.random.uniform(*gaussian_std_range)
    salt_prob     = np.random.uniform(*salt_prob_range)
    pepper_prob   = np.random.uniform(*pepper_prob_range)

    noise_field = np.zeros((H, W), dtype=np.float32)

    
    
    if np.random.rand() < gaussian_prob and gaussian_std > 0:
        noise_field += np.random.normal(gaussian_mean, gaussian_std, (H, W))

    img_noisy = img.copy()
    img_noisy += noise_field[:, :, None]  
    img_noisy = np.clip(img_noisy, 0, 255)


    if np.random.rand() < poisson_prob:
        value = np.random.randint(poisson_min, poisson_max + 1)
        gray = cv2.cvtColor(img_noisy.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lam = np.maximum(gray / 255.0 * value, 1e-6)
        samp = np.random.poisson(lam)
        factor = (samp / lam)  
        img_noisy *= factor[:, :, None]
        img_noisy = np.clip(img_noisy, 0, 255)

    
    if np.random.rand() < sp_prob:
        salt_mask   = (np.random.rand(H, W) < salt_prob)
        pepper_mask = (np.random.rand(H, W) < pepper_prob)
        img_noisy[salt_mask]   = 255
        img_noisy[pepper_mask] = 0
    
    out = img_noisy.astype(np.uint8)

    if return_params:
        max_img = img.copy().astype(np.float32)

        # Gaussian max
        if gaussian_std_range[1] > 0:
            max_img += np.random.normal(0, gaussian_std_range[1], (H, W))[:, :, None]
        max_img = np.clip(max_img, 0, 255)

        # Poisson max degradation -> use poisson_min
        if poisson_min > 0:
            gray_max = cv2.cvtColor(max_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            lam_max = np.maximum(gray_max / 255.0 * poisson_min, 1e-6)
            samp_max = np.random.poisson(lam_max)
            factor_max = (samp_max / lam_max)
            max_img *= factor_max[:, :, None]
            max_img = np.clip(max_img, 0, 255)

        # Salt & Pepper max
        salt_mask = (np.random.rand(H, W) < salt_prob_range[1])
        pepper_mask = (np.random.rand(H, W) < pepper_prob_range[1])
        max_img[salt_mask] = 255
        max_img[pepper_mask] = 0

        # L1-distance & degradation score
        l1_random = np.mean(np.abs(out.astype(np.float32) - img))
        l1_max = np.mean(np.abs(max_img.astype(np.float32) - img))
        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
      
        return out, deg_score
    return out



#----------------ir_stripe----------------------------------

def add_stripe_to_ir_image(img, beta_range=(3, 25),  return_params=False):
    
    beta = np.random.uniform(beta_range[0], beta_range[1])
  
    noise_col = np.random.normal(0, beta, img.shape[1])
    S_noise = np.tile(noise_col, (img.shape[0], 1))
    S_noise = np.stack([S_noise]*3, axis=2)  

  
    noisy_img = img.astype(np.float32) + S_noise

    
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    if return_params:
        beta_max = beta_range[1]
        noise_col_max = np.random.normal(0, beta_max, img.shape[1])
        S_noise_max = np.tile(noise_col_max, (img.shape[0], 1))
        S_noise_max = np.stack([S_noise_max]*3, axis=2)
        max_img = np.clip(img.astype(np.float32) + S_noise_max, 0, 255).astype(np.uint8)

        # L1-distance
        l1_random = np.mean(np.abs(noisy_img.astype(np.float32) - img.astype(np.float32)))
        l1_max = np.mean(np.abs(max_img.astype(np.float32) - img.astype(np.float32)))
        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
       
        return noisy_img, deg_score

    return noisy_img
    
 

#----------------ir_lowcontrast----------------------------------
    
def add_lowcontrast_to_ir_img(img, alpha_range=(0.3, 0.9), return_params=False):
    img = img.astype(np.float32)

    
    alpha = np.random.uniform(*alpha_range)
    beta  = 255 * (1 - alpha)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

  

    if return_params:
       
        alpha_max = alpha_range[0]
        beta_max = 255 * (1 - alpha_max)
        max_img = cv2.convertScaleAbs(img, alpha=alpha_max, beta=beta_max)

        l1_random = np.mean(np.abs(out.astype(np.float32) - img))
        l1_max = np.mean(np.abs(max_img.astype(np.float32) - img))
        deg_score = float(np.clip(l1_random / (l1_max + 1e-8), 0.0, 1.0))
        
        return out, deg_score

    return out

