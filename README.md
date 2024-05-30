# Proteus

[![standard-readme compliant](https://img.shields.io/badge/SE3%20StrcturePrediction%20-init-green.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)
[![standard-readme compliant](https://img.shields.io/badge/SE3%20ComplexDiffusion%20-init-green.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)
[![standard-readme compliant](https://img.shields.io/badge/SE3%20MoleculeDiffusion-Proposed-inactive.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)

Implementation for "Proteus: Pioneering Protein Structure Generation for Enhanced Designability and Efficiency".

If you use our work then please cite
```
@article {Wang2024proteus,
	author = {Wang, Chentong and Qu, Yannan and Peng, Zhangzhi and Wang, Yukai and Zhu, Hongli and Chen, Dachuan and Cao, Longxing},
	title = {Proteus: exploring protein structure generation for enhanced designability and efficiency},
	year = {2024},
	doi = {10.1101/2024.02.10.579791},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

## Table of Contents

- [Install](#install)
- [Inference](#inference)
- [License](#license)

## Install

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies. Using mamba if possible for better install speed.
```bash
# install
conda env create -f se3.yml
# optional : using mamba for faster environment installation
conda install mamba
mamba env create -f se3.yml

# activate environment
conda activate Proteus

## Inference


#### monomer inference(command used in paper)
For the first time run, it might be a little slow because of downloading esmfold ckpt 
```sh
python ./experiments/inference_se3_diffusion.py \
inference.output_dir=inference_outputs/monomer/ \
inference.weights_path=$weight_path \
inference.diffusion.samples.samples_lengths=[100,200,300,400,600,800] \
inference.diffusion.samples.samples_per_length=100 \
inference.diffusion.num_t=100

#To disable esmfold prediction and mpnn design, add extra config
inference.mpnn.enable=False inference.esmfold.enable=False

#To disable esmfold prediction add extra config
inference.esmfold.enable=False
```
A self_consistency.csv will be generated in the inference_outputs/monomer/${timestap}/self_consistency.csv, report all necessary metrics like dssp or sc-rmsd, etc.

#### oligomer inference
```sh
python ./experiments/inference_se3_diffusion.py \
inference.output_dir=inference_outputs/oligomer/ \
inference.weights_path=$baseline_weight_path \
inference.diffusion.samples.contigs='60-80//60-80' \
inference.diffusion.samples.samples_per_length=100 \
inference.diffusion.num_t=100
```

Inference output wuold be like
```shell
inference_outputs
└── 12D_02M_2023Y_20h_46m_13s           # Date time of inference.
    ├── mpnn.fasta                      # mpnn designed seuences.
    ├── self_consistency.csv            # self consistency analysis, contains rmsd and tmscore between scaffold ans esmfold, mpnn score of sequence, scaffold path, esmf path etc.
    ├── diffusion                       # dir contains scaffold generated by proteus
    │    ├── 100_1_sample.pdb          
    │    ├── 100_2_sample.pdb           # {length}_{sample_id}_sample.pdb
    |    └── ...
    ├── trajctory                       # dir contains traj pdb, exists when inference.diffusion.option.save_trajactory=True
    │    ├── 100_1_bb_traj.pdb          
    │    ├── 100_2_bb_traj.pdb          # {length}_{sample_id}_traj.pdb
    |    └── ...
    ├── movie                           # dir contains full atom protein designed by mpnn, exists when inference.diffusion.option.plot.switch_on=True
    │    ├── 100_1_rigid_movie.gif      # movie of protein rigid at time t    
    │    ├── 100_1_rigid_0_movie.gif    # movie of predict protein rigid at time 0 from time t  
    |    └── ...
    ├── mpnn                            # dir exists when pyrosetta in installed and inference.mpnn.dump=True
    │    ├── 100_0_sample_mpnn_0.pdb      
    │    ├── 100_0_sample_mpnn_1.pdb    # {length}_{sample_id}_sample_mpnn_{sequence_id}.pdb
    |    └── ... 
    └── esmf                            # dir contians esmf predict strcture
         ├── 100_0_sample_esmf_0.pdb     
         ├── 100_0_sample_esmf_0.pdb     # {length}_{sample_id}_sample_esmf_{sequence_id}.pdb
         └── ... 




## License

LICENSE: MIT
