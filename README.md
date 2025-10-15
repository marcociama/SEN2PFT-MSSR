
---

## Enel Project - RGB-NIR Super-Resolution

Modified version of PFT-SR adapted for 4-channel (RGB-NIR) satellite imagery super-resolution (4x upscaling).

### Key Modifications:
- Extended architecture for multispectral (RGB-NIR) input
- Custom dataset loader for satellite imagery
- Optimized training pipeline for remote sensing data
- Fine-tuned configurations for Enel's use case

### Project Structure:
- `basicsr/` - Modified BasicSR framework for RGB-NIR
- `experiments/PFT_RGBNIR_x4_v4/` - Main training experiment
- `experiments/PFT_RGBNIR_x4_v4_final_squeeze/` - Final optimized model
- `options/` - Training and testing configurations

---

## Pre-trained Models

**📥 Download Trained Models**: https://drive.google.com/drive/folders/1XGm8KkzYHRj51K4v_Y2621YInharXSos?usp=sharing

Due to file size limitations, trained model weights are hosted on Google Drive.

**Available Models:**
- Latest checkpoint and best performing models
- Place downloaded `.pth` files in `experiments/PFT_RGBNIR_x4_v4_final_squeeze/models/`

**Usage:**
```bash
python inference.py -i input_image.tif -o results/custom_location/ --scale 4 --task enel
```

---

## Credits

This project is based on the original PFT-SR repository.  
**Original License**: Apache 2.0  

All modifications for RGB-NIR satellite imagery adaptation are maintained under the Apache 2.0 license.

