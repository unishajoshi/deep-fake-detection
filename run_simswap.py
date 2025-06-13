# run_simswap.py

import argparse
import os
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap

def lcm(a, b): return abs(a * b) // fractions.gcd(a, b) if a and b else 0

# Set working directory to SimSwap
SIMSWAP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SIMSWAP_DIR)

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--source_image', type=str, required=True, help='Path to source image (face)')
    #parser.add_argument('--target_video', type=str, required=True, help='Path to target video')
    #parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated fake video')
    #parser.add_argument('--use_mask', action='store_true', help='Use mask during face swap')  # <-- add this line

    #args = parser.parse_args()

    # Load default SimSwap options
    opt = TestOptions().parse() 
    #opt.pic_a_path = args.source_image
    #opt.video_path = args.target_video
    #opt.output_path = args.output_path

    # Set default options
    arcface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "arcface_model", "arcface_checkpoint.tar"))
    #opt.Arc_path ="arcface_model/arcface_checkpoint.tar"
    opt.Arc_path = arcface_path
    opt.temp_path = "./temp_results"
    opt.no_simswaplogo = True
    #opt.use_mask = getattr(opt, 'use_mask', False)
    opt.crop_size = 512   # or 512 if needed
    opt.no_simswaplogo = True
    opt.use_mask = getattr(opt, 'use_mask', False)  # or False if needed
    opt.use_attention = getattr(opt, 'use_attention', False)

    try:
        source_img = Image.open(opt.pic_a_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to load source image: {opt.pic_a_path}. Error: {e}")

    if opt.crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    # Load SimSwap model
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    # Initialize face detector
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        img_a_whole = cv2.imread(opt.pic_a_path)
        result = app.get(img_a_whole, opt.crop_size)

        if result is None:
            raise ValueError(f"[ERROR] Face recognition failed for image: {opt.pic_a_path}")
        
        img_a_align_crop, _ = result
        
        img_a_align_crop, _ = app.get(img_a_whole, opt.crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        # Run face swapping
        video_swap(
            opt.video_path,
            latend_id,
            model,
            app,
            opt.output_path,
            temp_results_dir=opt.temp_path,
            no_simswaplogo=opt.no_simswaplogo,
            use_mask=opt.use_mask,
            crop_size=opt.crop_size
        )

if __name__ == '__main__':
    main()
