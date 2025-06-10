# BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly

![Overview](./img/teaser.jpg)

**(A)** Direct learning long-horizon action trajectories of geometric assembly may face many challenges: grasping ungraspable points, grasping points not suitable for assembly (e.g., seams of fragments), robot colliding with parts and the other robot. **(B)** We formulate this task into 3 steps: pick-up, alignment and assembly. For assembly, we predict the direction that will not result in part collisions. For alignment, we transformed any assembled poses to poses easy for the robot to manipulate from the initial poses without collisions. For pick-up, we learn point-level affordance aware of graspness and the following 2 steps. **(C)** Real-World Evaluations with affordance predictions on two mugs and the corresponding manipulation.



## About the Paper

BiAssambly is accepted to ICML 2025.

Arxiv Version: https://arxiv.org/pdf/2506.06221

Project Page: https://sites.google.com/view/biassembly/

## Before Start

To train the models, please first go to the `assets` directory and download the pre-processed Breaking-Bad dataset for BiAssembly and put it under `assets/object/breaking_bad`. More information about the Breaking-Bad dataset is available on this [website](https://breaking-bad-dataset.github.io).

To evaluate the pretrained models, please go to the `logs` directory and download the pretrained checkpoints.

For real-world benchmark, you can find both the reconstructions of complete shapes and the fractured parts in the `realworld-benchmark` directory.

You can click [here](https://mirrors.pku.edu.cn/dl-release/BiAssembly_ICML2025/) to download all resources.

## Dependencies

This code has been tested on Ubuntu 20.04 with Python 3.8 and PyTorch 1.6.0.

First, install [SAPIEN](**https://sapien.ucsd.edu**)：

    pip install sapien==2.2.2


Then, if you want to run the 3D experiment, this depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.

    # make sure you are at the repository root directory
    pip install -r requirements.txt

to install the other dependencies.

## Data Collection

Before training the network, it is necessary to collect a large set of interaction trials to serve as the training, validation, and testing datasets.

For example, to generate offline training data, run the following command:

    cd data_generation
    python collect_data.py --split_name Train

You can refer to `gen_cate_setting.py` to see the default settings of categories, modify the file and run this command to generate the json file for loading arguments:

    python gen_cate_setting.py

## Training Pipeline for the BiAssembly Framework

To train the **Disassembly Predictor** and the **Transformation Predictor**, run:

    cd method
    sh scripts/run_train_disassembly_and_transformation_predictor.sh

To train the **Bi-Affordance Predictor**, run:

    sh scripts/run_train_bi_affordance_predictor.sh

## Evaluation

To evaluate and visualize the results, run the following command:

    cd method
    sh scripts/run_inference_bi_assembly.sh

This script uses the pretrained networks to propose interactions in simulation. The manipulation results will be saved as `.gif` files. Additionally, affordance maps are visualized and saved in `.ply` format, which can be viewed using [MeshLab](https://www.meshlab.net/).

For convenience, we provide the testing dataset, which includes a set of novel instances within the training categories as well as a set of novel categories. It can be downloaded [here](https://mirrors.pku.edu.cn/dl-release/BiAssembly_ICML2025/). 

Please note that, due to the randomness in data collection, this provided testing dataset imay differ from those used in the paper, so you may result in slightly different but comparable results compared with the results in paper. Besides, since cVAE networks may produce varying outputs based on different Gaussian noise z sampled from a Gaussian distribution (depending on the random seed), we recommend setting the value of `repeat_num` to be greater than 1 to obtain more stable results.



## Citations

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@inproceedings{shen2025biassembly,
  title={BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly},
  author={Yan Shen and Ruihai Wu and Yubin Ke and Xinyuan Song and Zeyi Li and Xiaoqi Li and Hongwei Fan and Haoran Lu and Hao Dong},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2025},
}
```

## Questions

For questions or further assistance, please open an issue on our GitHub repository. We encourage using GitHub issues rather than email, as your questions may help others as well.

If you prefer, you can also contact us via email at: yan790 [at] pku [dot] edu [dot] cn.





