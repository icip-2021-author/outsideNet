# outsideNet - ICIP 2021 (#2587)
###
This is the code for the submission #2587 at ICIP 2021. 
We provide the pytorch implementation of our model, pretrained weights for the OUTSIDE15K data set and sample images for testing. 
## Run the code
### Requirements
Python: >3.2

CUDA: 10.2

cuDNN: 7.4.1

To install the required python packages you can run: 
```
pip install -r requirements.txt
```
### Quick start: 
Unzip the pretrained weights:
```
sh unzip_weights.sh
```

Test the network on the sample images:
```
python3 test.py --cfg config/outside15k-resnet50-outsideNet.yaml --imgs test_data/
```

Test the network on costum images or a folder of images::
```
python3 test.py --cfg config/outside15k-resnet50-outsideNet.yaml --imgs $PATH
```

The config file allows testing additional settings, e.g. multi-scale testing.
