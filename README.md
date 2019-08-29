# BigGAN by TensorFlow 2.0 RC
[BigGAN](https://arxiv.org/abs/1809.11096) implemented by tensorflow 2.0 RC version.

**!!!!!!!! NOT CONFIRMED TO WORK !!!!!!!!**

## Requirements
- numpy==1.17.0
- opencv-python==4.1.0.25
- scipy==1.3.1
- tensorboard==1.14.0
- tensorflow==2.0.0rc0
- tqdm==4.35.0

```
# Some related libraries will also be installed.
pip install -r requirements.txt
```

## Dataset preparation
The officially reported results are obtained by quite huge dataset, 
here I trained on a relatively small cat dataset with single GPU.

```
# Download cat dataset, this script finally creates "cats_bigger_than_128x128" directory.
sh setting_up_script.sh
```

## Let's train!
```
CUDA_VISIBLE_DEVICES=0 python train.py -d Cat -dd /path/to/cats_bigger_than_128x128 -i 100000 -b 32 --resize_shape 128 128 --flip_lr
```

`-d` and `-dd` indicate the target dataset name and the corresponding directory.
Other configs are stated in `utils.py`. Hyperparameters are almost the same as the paper.

Training log can be watched in TensorBoard.
```
tensorboard --logdir=logs
```

## Other dataset?
You can try to train on the other dataset by inherit the `Base` dataset class in `datasets.py`.

## Acknowledgments
- Official TensorFlow documents: https://www.tensorflow.org/beta
- The author's PyTorch implementation of BigGAN: https://github.com/ajbrock/BigGAN-PyTorch
- Using Cat dataset for GAN: https://github.com/AlexiaJM/Deep-learning-with-cats
- Sophisticated solution in Kaggle competition: https://github.com/bestfitting/kaggle/tree/master/gandogs
- Easy-to-read tensorflow implementation of BigGAN: https://github.com/taki0112/BigGAN-Tensorflow
