## Image super-resolution with convolutional neural networks

Written by Caleb Zulawski and Kelvin Lin for ECE411 *Computational Graphs for Machine Learning* at The Cooper Union.
The implementation is based on [*Image Super-Resolution Using Deep
Convolutional Networks*](https://arxiv.org/pdf/1501.00092v3.pdf).

---

## Dataset
Images are downloaded from ImageNet from a variety of categories.  Subimages (33x33) are randomly cropped from each image and used to train the network.

## Convolutional Neural Network
The network is designed to improve the perceived quality of an image upscaled with bicubic interpolation.  The network is made up of 3 convolutional layers (9x9, 64 features; 1x1, 32 features; 5x5, 3 channels).  Padding is not used, so the output image is 12 pixels smaller in each dimension.
#### Training
To train the network, a Gaussian blur is applied to each subimage (3x3 kernel, σ=0.2), before downsampling by 3 and interpolating by 3 using bicubic interpolation.  This interpolated subimage is the input to the network, and the output is compared with the original full-resolution image.  The results are validated by calculating the peak signal-to-noise ratio (PSNR) gain from the bicubic interpolation to the output of the CNN.  When we stopped our training, the PSNR gain was approximately 1.2 dB. 
#### Generation
When generating new images at a higher resolution, the image is scaled by 3 using bicubic interpolation and input to the network.

## Downloading ImageNet
To download ImageNet, you'll need an account and access to the original ImageNet data, which is freely available for educational and non-commercial use.  This will provide you with an API access key, which is needed to download the data.

Once you have an access key, create a file named `imagenet_credentials.sh` which contains your credentials in the following format:

```bash
username=yourusername
accesskey=youraccesskey
```

Then you can run `./get_data.sh` which will download and prepare the images from the ImageNet synsets specified in `wnids.txt`.

## Example Images

[01-input]: img/01-input.jpg
[01-bicubic]: img/01-bicubic.jpg
[01-output]: img/01-output.jpg
[02-input]: img/02-input.jpg
[02-bicubic]: img/02-bicubic.jpg
[02-output]: img/02-output.jpg
[03-input]: img/03-input.jpg
[03-bicubic]: img/03-bicubic.jpg
[03-output]: img/03-output.jpg


| Original    | Bicubic       | Network      |
| ----------- | ------------- | ------------ |
| ![01-input] | ![01-bicubic] | ![01-output] |
| ![02-input] | ![02-bicubic] | ![02-output] |
| ![03-input] | ![03-bicubic] | ![03-output] |
