# Sample Select Policy
Which images to label for few-shot medical landmark detection? 




## Environment

```
python == 3.5/3.6, 
pytorch >= 1.1.0, 
torchvison >= 0.6
```

## Data preparation

We train/test our model on Cephalometric Dataset

We expect the directory structure to be the following:

```
path/to/cephalometric
	400_junior
		001.txt
		...
	400_senior
		001.txt
		...
	RawImage
		TrainingData
			001.bmp
			...
		Test1Data
			151.bmp
			...
		Test2Data
			301.bmp
			...
```

## Steps

1. Train the feature extractor
```python
python -m sc.ssl.ssl --tag run
```
2. extract SIFT key points
```python
python -m sc.select.sift_select --tag sift
```

3. calculate similarities (Respective score)
```python
python -m sc.select.maxsim_sift --tag sift
```

4. select templates

```
python -m sc.select.selct_ids --tag sift 
```

### Evaluation
1. Estimate all MRE
```
python -m sc.select.test_by_multi --tag xx
```
2. Test templates
```
python -m sc.select.test_specific_ids --indices xxx
```

# License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

```
@InProceedings{Quan_2022_CVPR,
    author    = {Quan, Quan and Yao, Qingsong and Li, Jun and Zhou, S. Kevin},
    title     = {Which Images To Label for Few-Shot Medical Landmark Detection?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {20606-20616}
}
```


   
