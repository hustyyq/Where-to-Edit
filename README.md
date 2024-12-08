# Learning Where to Edit Vision Transformers
NeurIPS 2024: [Learning Where to Edit Vision Transformers](https://arxiv.org/abs/2411.01948) (Pytorch implementation).  

### Constructed Editing Benchmark
- Underrepresented Natural and AI-generated Images (Evaluating Reliability and Generalization)
    - Natural Image Subset: The images are available at [DropBox Link](https://www.dropbox.com/scl/fi/5os0w5ii2f8e7l77au4wt/Natural-Image-Subset.zip?rlkey=jikug6el8ghkz7oiewaplum61&st=uj65rbfq&dl=0).
    - AI Oil Painting: Please refer to [Dataset Interfaces](https://github.com/MadryLab/dataset-interfaces).
    - AI Stage Lighting: Please refer to [PUG](https://github.com/facebookresearch/PUG).
- Locality Set
    - A sensitive subset from [ImageNet-1k](http://image-net.org/challenges/LSVRC/2012/index), [ImageNet-R](https://github.com/hendrycks/imagenet-r), and [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch). The  selection criteria rely on the predicted probabilities of the pre-trained ViT/B-16 model as follows: a. the predicted probability for the true label is the highest, and 2) the differencebetween the top two predicted probabilities is less than 0.05, suggesting a highly ambiguous class.
    - The sensitive subset for ViT/B-16 is provided at [DropBox Link](https://www.dropbox.com/scl/fi/0zgd2p2ya3p67c7wqy3mo/Sensitive-Images.zip?rlkey=tpfbo4br1qkowjj8phlj2vz9w&st=ihbgsda1&dl=0).

### Prepare Datasets and Pre-trained Models
 - Download pre-trained [ViT/B-16](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) and revise the model path in line 542 of models/vit.py
 - Download the Natural-Image-Subset and the sensitive subset for ViT/B-16.


### Meta-training the Hypernetwork
```
python meta_train.py -a vit_b_16_224_1k --pre_root [imagenet1k dataset dir] --lr 1e-4 --elr 1e-3 --mes 5 -b 8 --blocks 3
```

### Editing Models
- FT
    ```
    python  edit_natural.py --root ./Natural-Image-Subset --seed 0 -a vit_b_16_224_1k   --edit-lrs 2e-5  --alg HPRD  --log  ./log/natural/FT  --max-iters 100  
    ```
- HPRD
    ```
    python  edit_natural.py --root ./Natural-Image-Subset  --seed 0 -a vit_b_16_224_1k   --edit-lrs 2e-5  --alg HPRD  --log  ./log/natural/HPRD  --max-iters 100 --checkpoints ./logs/checkpoints/7000.pth --blocks 3
    ```
### Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{yang2024learning,
    title={Learning Where to Edit Vision Transformers},
    author={Yunqiao Yang and Long-Kai Huang and Shengzhuang Chen and Kede Ma and Ying Wei},
    booktitle={Neural Information Processing Systems},
    year={2024}
}
```
### Acknowledgements
Thank the Pytorch implementation of Vision transformers in [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch).