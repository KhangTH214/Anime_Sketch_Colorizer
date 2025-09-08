# Anime_Sketch_Colorizer
Automatic Sketch Colorization with reference image
## Prerequisites
<table> <tr> <td align="center" width="150"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" width="40" height="40"/> <br/> <strong>PyTorch</strong> <br/> <a href="https://pytorch.org">Docs</a> </td> <td align="center" width="150"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/torch/torch-original.svg" width="40" height="40"/> <br/> <strong>TorchVision</strong> <br/> <a href="https://pytorch.org/vision">Docs</a> </td> <td align="center" width="150"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="40" height="40"/> <br/> <strong>NumPy</strong> <br/> <a href="https://numpy.org">Docs</a> </td> </tr> <tr> <td align="center" width="150"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" width="40" height="40"/> <br/> <strong>OpenCV</strong> <br/> <a href="https://opencv.org">Docs</a> </td> <td align="center" width="150"> <img src="https://matplotlib.org/stable/_static/logo2_compressed.svg" width="40" height="40"/> <br/> <strong>Matplotlib</strong> <br/> <a href="https://matplotlib.org">Docs</a> </td> </tr> </table>

## Dataset
Taebum Kim, <a href="https://www.kaggle.com/ttaebum/anime-sketch-colorization-pair">Anime Sketch Colorization Pair</a>, Kaggle
## Train
Please refer to the training notebook: train.ipynb
## Training details

| Parameter                | Value                                        |
|--------------------------|----------------------------------------------|
| Learning rate            | 2e-4                                         |
| Batch size               | 2                                            |
| Epoch                    | 25                                           |
| Optimizer                | Adam                                         |
| (beta1, beta2)           | (0.5, 0.999)                                |
| (lambda1, lambda2, lambda3) | (100, 1e-4, 1e-2)                        |
| Data Augmentation        | RandomResizedCrop(256) <br> RandomHorizontalFlip() |
| HW                       | CPU : Intel i5-8400 <br> RAM : 16G <br> GPU : NVIDIA GTX1060 6G |
| Training Time            | About 0.93s per iteration <br> (About 45 hours for 25 epoch) |
## Modal
<img width="2220" height="1049" alt="Modal" src="https://github.com/user-attachments/assets/2aa7d131-a596-4051-b580-3e49c9ece16b" />
## Results
![img_1](https://github.com/user-attachments/assets/00a76a64-1695-4182-8aa8-56fb05861c08)
![img_2](https://github.com/user-attachments/assets/6f37e37e-65bc-4dad-ac17-3c5a00aac3ca)
![img_3](https://github.com/user-attachments/assets/4619990d-70d5-4ae3-b561-067cd40dcccb)
![img_4](https://github.com/user-attachments/assets/def23dfc-3a31-45dd-bf25-fad5061fefd3)

