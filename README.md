This is the project page of our paper:  

**Universal Perturbation Attack Against Image Retrieval**,  
Li, J., Ji, R., Liu, H., Hong, X., Gao, Y., & Tian, Q. 
ICCV 2019.
[[PDF]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Universal_Perturbation_Attack_Against_Image_Retrieval_ICCV_2019_paper.pdf)

## Code
Our codes are based on [filipradenovic/cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (Commit `c4fca89`).
Please clone their repository first, and then put the `attack` folder here in `cirtorh/examples/`

### Prepare Features
1. Follow the steps in [filipradenovic/cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) to download datasets and train the retrieval models.
2. Refer to the function `cluster()` in `attack/myutil/triplet_dataset.py` about extracting features and clustering.

### Train Classifiers (Optional)
```
python -m cirtorch.examples.attack.classifier.py PATH
```

### Generate UAP
Refer to arguments in `attack/attack.py` for details.

### Ranking Distillation
1. Refer to `attack/extract_rank.py` for extracting ranking list.
2. Refer to `attack/distillation.py` for distillation.


## Typos in Paper
1. Eq. 6 should be ![](http://latex.codecogs.com/gif.latex?\\frac{\\partial%20d(f,f_j)}{\\partial\\delta}-\\frac{\\partial%20d(f,f_k)}{\\partial\\delta})
2. Eq. 7 should be ![](http://latex.codecogs.com/gif.latex?m<n})

I'm sorry for typos in this paper. If you find more typos, please do not hesitate to point out in [issues](https://github.com/theFool32/UAP_retrieval/issues).

## Citation  
If our paper helps your research, please cite it in your publications:
```
@InProceedings{Li_2019_ICCV,
author = {Li, Jie and Ji, Rongrong and Liu, Hong and Hong, Xiaopeng and Gao, Yue and Tian, Qi},
title = {Universal Perturbation Attack Against Image Retrieval},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

Feel free to contact to the authors (lijie.32@outlook.com) or create a new issue if you find any problems.
