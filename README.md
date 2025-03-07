# AM-UNET_reproduction_tensorflow
This repository using tensorflow to reproduce AM-UNET model . 

Reference of the original GitHub: https://github.com/AhmedAlbishri/AM-UNET

Reference of the original paper: DOI: 10.1007/s11042-021-11568-7


## Overview
```bibtex
@article{albishri2022unet,
  title={AM-UNet: automated mini 3D end-to-end U-net based network for brain claustrum segmentation},
  author={Albishri, Ahmed Awad and Shah, Syed Jawad Hussain and Kang, Seung Suk and Lee, Yugyung},
  journal={Multimedia Tools and Applications},
  volume={81},
  number={25},
  pages={36171--36194},
  year={2022},
  publisher={Springer}
}
```


## Environments and Requirements

This test implementation is designed to run on **CPU**.

**Python**: Version 3.10

To set up the environment:
```bash
git clone https://github.com/ShutingXie/AM-UNET_reproduction_tensorflow.git
```

Create a virtual environment (You can use other way to creat a virtual environment):
```bash
python -m venv myenv
```

Activate the virtual environment:
```bash
source myenv/bin/activate
```

Install all dependent packages
```bash
pip install -r requirements.txt
```


## Dataset
Put your MRI data and labels in the **input_data/** folder


## Test
```bash
cd use_h5
python main.py
```

## Evaluation the prediction results
```bash
python compute_agreement.py
```





   
