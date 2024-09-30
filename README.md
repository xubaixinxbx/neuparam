<h1 align="center">Parameterization-driven Neural Surface Reconstruction for Object-oriented Editing in Neural Rendering
</h1>
  <p align="center">
    <a href="https://xubaixinxbx.github.io/">Baixin Xu</a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=6TG39EcAAAAJ&hl">Jiangbei Hu</a>
    ·
    <a href="https://lcs.ios.ac.cn/~houf/">Fei Hou</a>
    .
    <a href="https://kwanyeelin.github.io/">Kwan-Yee Lin</a>
    .
    <a href="https://wywu.github.io/">Wayne Wu</a>
    .
    <a href="https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en">Chen Qian</a>
    .
    <a href="https://personal.ntu.edu.sg/yhe/">Ying He</a>
  </p>
  <h3 align="center">ECCV 2024</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2310.05524">Paper</a> | <a href="https://xubaixinxbx.github.io/neuparam/">Project Page</a></h3>
  <div align="center"></div>
</p>

## Setup
### Installation

```
conda env create -f environment.yml
conda activate neuparam
```

### Dataset
[OmniObject3D](https://omniobject3d.github.io/), [FaceScape](https://facescape.nju.edu.cn/)

### Training and inference

```
cd code
bash confs/omni/train_omni.sh
```
## Citation

If you find our work useful, please kindly cite as:
```
@inproceedings{xu2024neuparam,
    title={Parameterization-driven Neural Surface Reconstruction for Object-oriented Editing in Neural Rendering},
    author={Xu, Baixin and Hu, Jiangbei and Hou, Fei and Lin, Kwan-Yee and Wu, Wayne and Qian, Chen and He, Ying},
    booktitle={ECCV},
    year={2024}
    }
```

## Acknowledgement
* The codebase is developed based on [VolSDF](https://github.com/lioryariv/volsdf) of Yariv et al. We also acknowledge [NeuTex](https://github.com/fbxiang/NeuTex). Many thanks to their great contributions!