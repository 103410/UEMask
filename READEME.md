# UEMask: High-Fidelity Unlearnable Examples for Privacy Protection in Personalized Diffusion Models

This repository implements **UEMask**, 
a framework designed to protect facial identities from unauthorized subject-driven text-to-image generation (specifically DreamBooth). 
The pipeline generates protected images (unlearnable examples) that, when used for training, prevent the model from learning the target identity.



## 📋 Table of Contents
- [Environments setup](#requirements)
- [Dataset preparation](#directory-structure)
- [How to run](#usage-pipeline)
    


---

##  Environments setup

### Python Dependencies
Install the required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```
Pretrained checkpoints of  Stable Diffusion 2-1-base version can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/Manojb/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  
</table>

Please put them in `./model/`. Note: Stable Diffusion version 2.1 is the default version in all of our experiments.
## Dataset preparation
We utilize the pre-processed versions of the CelebA-HQ 
and VGGFace2 datasets provided by Anti-DB. 
This benchmark selects 50 identities from each dataset, with 12 high-quality images per identity. 
Following the standard protocol, 
these images are partitioned into three subsets: Set A, Set B, and Set C. In our experiments, 
we specifically select Set B as the target set for generating protected images (unlearnable examples). 
These full datasets are provided at [here](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV).
## How to run
### Protected image generation
```bash
bash ./scripts/train_uemask.sh
```
### Train DreamBooth model
```bash
bash ./scripts/train_dreambooth_multi.sh
```
### Infer
```bash
bash ./scripts/infer.sh
```
### Evaluation
We provide several scripts to evaluate both the visual quality of the protected images and the effectiveness of the protection.

##### BRISQUE

```bash
bash evaluation_brisque.sh
```
#### ISM
```bash
bash ./scripts/evaluation_ism.sh
```
#### SER-FIQ
```bash
python ./evaluation/ser_fiq.py
```
#### LPIPS
```bash
python ./visual_quality_evaluation/lpips.py
```
#### SSIM
```bash
python ./visual_quality_evaluation/ssim.py
```

