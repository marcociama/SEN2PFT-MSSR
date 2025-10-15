import torch
import os
import os.path as osp
import sys
import argparse
import tifffile
import numpy as np

from PIL import Image
from torchvision import transforms
from basicsr.archs.pft_arch import PFT

model_path = {
    "enel": {
        "4": "experiments/PFT_RGBNIR_x4_v4_final_squeeze/models/net_g_80000.pth"
    },
    "classical": {
        "2": "experiments/pretrained_models/001_PFT_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/002_PFT_SRx3_finetune.pth",
        "4": "experiments/PFT_RGBNIR_x4_v3/models/net_g_100000.pth"
    },
    "lightweight": {
        "2": "experiments/pretrained_models/101_PFT_light_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/102_PFT_light_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/103_PFT_light_SRx4_finetune.pth",
    }
}

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input image or directory path.")
    parser.add_argument("-o", "--out_path", type=str, default="results/test/", help="Output directory path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
            "--task",
            type=str,
            default="classical",
            choices=['classical', 'lightweight','enel'],
            help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
            )
    args = parser.parse_args()

    return args


"""
# original
def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert('RGB')
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)
        image_output = model(image_input).clamp(0.0, 1.0)[0].cpu()
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)"""

def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        if image_input_path.endswith('.tif') or image_input_path.endswith('.tiff'):
            # Carica immagine multispettrale già normalizzata (128,128,4)
            image_array = tifffile.imread(image_input_path)  # shape: (H, W, 4)
            ### SOTTRAZIONE MEDIA
            mean = np.array([0.3509425779495244, 0.34561040795129644, 0.3249112233408325, 0.42880553175091063], dtype=np.float32)
            image_array = image_array - mean
            if image_array.shape[-1] != 4:
                raise ValueError(f"Expected 4 channels, got {image_array.shape[-1]}")

            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # → (C, H, W)
        elif image_input_path.endswith('.npy'):
            image_array = np.load(image_input_path)
            if image_array.shape[-1] != 4:
                raise ValueError(f"Expected 4 channels, got {image_array.shape[-1]}")

            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        else:
            raise ValueError("This script is now only set to handle .tif images with 4 channels")

        image_tensor = image_tensor.unsqueeze(0).to(device)  # → (1, 4, H, W)
        output_tensor = model(image_tensor)[0]  # (4, H, W)
        
        # DENORMALIZZA (ancora come tensor, prima del clamp!)
        mean_tensor = torch.from_numpy(mean).view(4, 1, 1).to(output_tensor.device)
        output_tensor = output_tensor + mean_tensor
        
        # ORA fa il clamp e converti a numpy
        output_tensor = output_tensor.clamp(0.0, 1.0).cpu()
        output_numpy = output_tensor.permute(1, 2, 0).numpy()  # (H, W, 4)
        tifffile.imwrite(os.path.splitext(image_output_path)[0] + '.tif', output_numpy.astype('float32'))

def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task == 'classical':
        model = PFT(
                    in_chans=4, ### VERSIONE RGB + NIR
                    upscale=args.scale,
                    embed_dim=240,
                    depths=[4, 4, 4, 6, 6, 6],
                    num_heads=6,
                    num_topk=[1024, 1024, 1024, 1024,
                             256, 256, 256, 256,
                             128, 128, 128, 128,
                             64, 64, 64, 64, 64, 64,
                             32, 32, 32, 32, 32, 32,
                             16, 16, 16, 16, 16, 16, ],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    use_checkpoint=False,
                    )
        
    elif args.task == 'enel':
        model = PFT(
                    in_chans=4, ### VERSIONE RGB + NIR
                    upscale=args.scale,
                    embed_dim=240,
                    depths=[4, 4, 4, 6, 6, 6],
                    num_heads=6,
                    num_topk=[1024, 1024, 1024, 1024,
                             256, 256, 256, 256,
                             128, 128, 128, 128,
                             64, 64, 64, 64, 64, 64,
                             32, 32, 32, 32, 32, 32,
                             16, 16, 16, 16, 16, 16, ],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    use_checkpoint=False,
                    )
        
    elif args.task == 'lightweight':
        model = PFT(
                    in_chans=4, ### VERSIONE RGB + NIR (lightweight)
                    upscale=args.scale,
                    embed_dim=52,
                    depths=[2, 4, 6, 6, 6],
                    num_heads=4,
                    num_topk=[1024, 1024,
                              256, 256, 256, 256,
                              128, 128, 128, 128, 128, 128,
                              64, 64, 64, 64, 64, 64,
                              32, 32, 32, 32, 32, 32],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=1,
                    upsampler='pixelshuffledirect',
                    use_checkpoint=False,
                    )

    """
    # original
    state_dict = torch.load(model_path[args.task][str(args.scale)], map_location=device)['params_ema']
    model.load_state_dict(state_dict, strict=True)"""

    # OUR VERSION
    checkpoint = torch.load(model_path[args.task][str(args.scale)], map_location=device)
    state_dict = checkpoint['params_ema']

    # Adattamento per in_chans=4 e out_chans=4 se necessario

    # 1) conv_first (da 3 a 4 canali in input)
    if model.conv_first.weight.shape[1] == 4 and state_dict['conv_first.weight'].shape[1] == 3:
        print("Adatto conv_first da 3 a 4 canali")
        old_w = state_dict['conv_first.weight']   # (embed_dim, 3, 3, 3)
        new_w = torch.zeros((old_w.shape[0], 4, 3, 3), dtype=old_w.dtype)
        new_w[:, :3, :, :] = old_w
        new_w[:, 3, :, :] = old_w[:, 1, :, :]  # riciclo il canale G come NIR
        state_dict['conv_first.weight'] = new_w

    # 2) conv_last.weight (da 3 a 4 canali in output)
    if model.conv_last.weight.shape[0] == 4 and state_dict['conv_last.weight'].shape[0] == 3:
        print("Adatto conv_last da 3 a 4 canali")
        old_w_last = state_dict['conv_last.weight']
        feat = old_w_last.shape[1]
        new_w_last = torch.zeros((4, feat, 3, 3), dtype=old_w_last.dtype)
        new_w_last[:3] = old_w_last
        new_w_last[3] = old_w_last[1]  # riciclo G per NIR
        state_dict['conv_last.weight'] = new_w_last

    # 3) conv_last.bias (da 3 a 4 valori)
    if model.conv_last.bias.shape[0] == 4 and state_dict['conv_last.bias'].shape[0] == 3:
        print("Adatto conv_last.bias da 3 a 4 valori")
        old_b_last = state_dict['conv_last.bias']
        new_b_last = torch.zeros(4, dtype=old_b_last.dtype)
        new_b_last[:3] = old_b_last
        new_b_last[3] = old_b_last[1]
        state_dict['conv_last.bias'] = new_b_last


    model.load_state_dict(state_dict, strict=False)
    # END OUR VERSION
  
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff','.npy')):
                image_input_path = osp.join(args.in_path, file)
                file_name = osp.splitext(file)
                image_output_path = os.path.join(args.out_path, file_name[0] + '_PFT_' + args.task + '_SRx' + str(args.scale) + file_name[1])
                process_image(image_input_path, image_output_path, model, device)
    else:
        if args.in_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff','.npy')):
            image_input_path = args.in_path
            file_name = osp.splitext(osp.basename(args.in_path))
            image_output_path = os.path.join(args.out_path, file_name[0] + '_PFT_' + args.task + '_SRx' + str(args.scale) + file_name[1])
            process_image(image_input_path, image_output_path, model, device)


if __name__ == "__main__":
    main()
