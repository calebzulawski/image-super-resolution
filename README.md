## Image super-resolution with convolutional neural networks

Written by Caleb Zulawski and Kelvin Lin for ECE411 *Computational Graphs for Machine Learning* at The Cooper Union.

---


## Downloading ImageNet
To download ImageNet, you'll need an account and access to the original ImageNet data, which is freely available for educational and non-commercial use.  This will provide you with an API access key, which is needed to download the data.

Once you have an access key, create a file named `imagenet_credentials.sh` which contains your credentials in the following format:

```bash
username=yourusername
accesskey=youraccesskey
```

Then you can run `./get_data.sh` which will download and prepare the images from the ImageNet synsets specified in `wnids.txt`.
