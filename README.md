# [ICCV 2023]. Learning Optical Flow from Event Camera with Rendered Dataset. [[Paper]](https://arxiv.org/abs/2303.11011).
<h4 align="center">Xionglong Luo<sup>1,3</sup>, Kunming Luo<sup>2</sup>, Ao Luo<sup>3</sup>, Zhengning Wang<sup>1</sup>, Ping Tan<sup>2</sup>, Shuaicheng Liu<sup>1,3</sup></center>
<h4 align="center">1.University of Electronic Science and Technology of China
<h4 align="center">2.The Hong Kong University of Science and Technology, 3.Megvii Technology </center></center>

## Pipeline
<img src="assets/Datapipeline.png" width="1000">

## Environments
You will have to choose cudatoolkit version to match your compute environment. The code is tested on Python 3.7 and PyTorch 1.10.1+cu113 but other versions might also work. 
```bash
conda create -n admflow python=3.7
conda activate admflow
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements
```
## Dataset
### MVSEC
You need download the HDF5 files version of [MVSEC](https://daniilidis-group.github.io/mvsec/download/) datasets. We provide the code to encode the events and flow label of MVSEC dataset.
```python
# Encoding Events and flow label in dt1 setting
python loader/MVSEC_encoder.py --only_event -dt=1
# Encoding Events and flow label in dt4 setting
python loader/MVSEC_encoder.py --only_event -dt=4
# Encoding only Events
python loader/MVSEC_encoder.py --only_event
```
The final tree structure of MVSEC dataset should be：
```
dataset/MVSEC
├── indoor_flying1
│   ├── event
│   ├── flowgt_dt1
│   ├── flowgt_dt4
├── indoor_flying2
│   ├── event
│   ├── flowgt_dt1
│   ├── flowgt_dt4
├── indoor_flying3
│   ├── event
│   ├── flowgt_dt1
│   ├── flowgt_dt4
├── outdoor_day1
│   ├── event
│   ├── flowgt_dt1
│   ├── flowgt_dt4
├── outdoor_day2
│   ├── event
│   ├── flowgt_dt1
│   ├── flowgt_dt4
```
### MDR
This work proposed a Multi Density Rendered (MDR) event optical flow dataset, you can download it from https://pan.baidu.com/s/1iSgGCjDask-M_QqPRtaLhA?pwd=z52j . We also provide code for batch organizing MDR datasets.
```python
python loader/MDR_menage.py -dt=1
python loader/MDR_menage.py -dt=4
```
The final tree structure of MDR dataset should be：
```
dataset/MDR
├── dt1
│   ├── train
│   │   ├── best_density_events1
│   │   ├── best_density_events2
│   │   ├── events1
│   │   ├── events2
│   │   ├── flow
│   ├── test
│   │   ├── 0.09_0.24
│   │   │   ├── events1
│   │   │   ├── events2
│   │   │   ├── flow
│   │   ├── 0.24_0.39
│   │   │   ├── events1
│   │   │   ├── events2
│   │   │   ├── flow
│   │   ├── 0.39_0.54
│   │   │   ├── events1
│   │   │   ├── events2
│   │   │   ├── flow
│   │   ├── 0.54_0.69
│   │   │   ├── events1
│   │   │   ├── events2
│   │   │   ├── flow
```
## Evaluate
### Pretrained Weights
Pretrained weights can be downloaded from 
[Google Drive](https://drive.google.com/drive/folders/15uwhrmUzg3kK3UB6z0Qnht-sGs7Nq23o?usp=sharing).
Please put them into the `checkpoint` folder.

### Test on MVSEC
```python
# Dense evaluation
python test_mvsec.py -dt dt1
python test_mvsec.py -dt dt4
# Sparse evaluation
python test_mvsec.py -dt dt1 -eval
python test_mvsec.py -dt dt4 -eval
```

### Test on MDR
```python
# Dense evaluation
python test_mdr.py -dt dt1
python test_mdr.py -dt dt4
# Sparse evaluation
python test_mdr.py -dt dt1 -eval
python test_mdr.py -dt dt4 -eval
```
## Citation

If this work is helpful to you, please cite:

```
@article{luo2023learning,
  title={Learning Optical Flow from Event Camera with Rendered Dataset},
  author={Luo, Xinglong and Luo, Kunming and Luo, Ao and Wang, Zhengning and Tan, Ping and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2303.11011},
  year={2023}
}
```
## Acknowledgments

Thanks the assiciate editor and the reviewers for their comments, which is very helpful to improve our paper. 

Thanks for the following helpful open source projects:

[ERAFT](https://github.com/uzh-rpg/E-RAFT), 
[STE-FlowNet](https://github.com/ruizhao26/STE-FlowNet/),
[v2e](https://github.com/SensorsINI/v2e),
[KPAFlow](https://github.com/megvii-research/KPAFlow).

