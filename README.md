## Image super-resolution with convolutional neural networks

Written by Caleb Zulawski and Kelvin Lin for ECE411 *Computational Graphs for Machine Learning* at The Cooper Union.
The implementation is based on [*Image Super-Resolution Using Deep
Convolutional Networks*](https://arxiv.org/pdf/1501.00092v3.pdf).

---


## Downloading ImageNet
To download ImageNet, you'll need an account and access to the original ImageNet data, which is freely available for educational and non-commercial use.  This will provide you with an API access key, which is needed to download the data.

Once you have an access key, create a file named `imagenet_credentials.sh` which contains your credentials in the following format:

```bash
username=yourusername
accesskey=youraccesskey
```

Then you can run `./get_data.sh` which will download and prepare the images from the ImageNet synsets specified in `wnids.txt`.

## Images

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
