# Hybrid Frequency Modulation Network for Image Restoration [IJCAI'24]

[supplementary material, visual results, and models can be found here](https://drive.google.com/drive/folders/1qOkuV3jNBcqIZpFi1TfjJht15gHN9AnD?usp=sharing)

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

Computational complexity:
for dehazing 41.19 GFLOPs, 4.27M


## Citation
~~~
@inproceedings{cui2024hybrid,
  title={Hybrid Frequency Modulation Network for Image Restoration},
  author={Cui, Yuning and Liu, Mingyu and Ren, Wenqi and Knoll, Alois},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2024}
}
~~~


## Contact
Should you have any question, please contact Yuning Cui.
