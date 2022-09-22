# TransVG

此仓库基于[djiajunustc/TransVG (github.com)](https://github.com/djiajunustc/TransVG)官方版本修改，by Hao Bian

## 复现

1. 首先按照官方教程配置环境, 数据划分的文件需放在`CODE_ROOT/data`

   ```bash
   CODE_ROOT/data
   ├── flickr
   │   ├── corpus.pth
   │   ├── flickr_test.pth
   │   ├── flickr_train.pth
   │   └── flickr_val.pth
   ├── gref
   │   ├── corpus.pth
   │   ├── gref_train.pth
   │   └── gref_val.pth
   ├── gref_umd
   │   ├── corpus.pth
   │   ├── gref_umd_test.pth
   │   ├── gref_umd_train.pth
   │   └── gref_umd_val.pth
   ├── referit
   │   ├── corpus.pth
   │   ├── referit_test.pth
   │   ├── referit_train.pth
   │   ├── referit_trainval.pth
   │   └── referit_val.pth
   ├── unc
   │   ├── corpus.pth
   │   ├── unc_testA.pth
   │   ├── unc_testB.pth
   │   ├── unc_train.pth
   │   ├── unc_trainval.pth
   │   └── unc_val.pth
   └── unc+
       ├── corpus.pth
       ├── unc+_testA.pth
       ├── unc+_testB.pth
       ├── unc+_train.pth
       ├── unc+_trainval.pth
       └── unc+_val.pth
   ```

2. 原数据按如下格式放置在`CODE_ROOT/ln_data`, (可建立软链接，如`ln -s src_data_path CODE_ROOT/ln_data`, 源文件数据在公共目录`/cto_studio/datastory/phrase_grounding/dataset`)

   ```bash
   ln_data/
   ├── data.tar
   ├── MSCOCO
   │   ├── train2014
   ├── RefCOCO
   │   ├── refcoco
   │   │   ├── instances.json
   │   │   ├── refs(google).p
   │   │   └── refs(unc).p
   │   ├── refcoco+
   │   │   ├── instances.json
   │   │   └── refs(unc).p
   │   └── refcocog
   │       ├── instances.json
   │       ├── refs(google).p
   │       └── refs(umd).p
   └── Flickr
   └── Flickr_Entities
   └── VG
   └── ZSG
   
   
   ```

3. 放置detr的预训练模型，在`CODE_ROOT/checkpoints`

   ```bash
   checkpoints/
   ├── detr-r50-referit.pth
   ├── detr-r50-unc.pth
   └── download_detr_model.sh
   ```

   

4. 写了一个脚本，方便不同数据集统一训练

```bash
GPUS=0 # 指定设备
DATASET=refcoco # refcoco+, refcocog_g, refcocog_u
sh train_dataset.sh $GPUS $DATASET
```

5. 写了一个脚本，方便不同数据集统一测试

```bash
GPUS=0 # 指定设备
DATASET=refcoco # refcoco+, refcocog_g, refcocog_u
sh test_dataset.sh $GPUS $DATASET
```

6. 复现结果文件在`CODE_ROOT/outputs`， 如refcoco数据集

|                     | val                | testA              |
| ------------------- | ------------------ | ------------------ |
| refcoco（ResNet50） | 0.8118081180811808 | 0.8252032520325203 |



### Installation
1.  Clone this repository.
    ```
    git clone https://github.com/djiajunustc/TransVG
    ```

2.  Prepare for the running environment. 

    You can either use the docker image we provide, or follow the installation steps in [`ReSC`](https://github.com/zyang-ur/ReSC). 

    ```
    docker pull djiajun1206/vg:pytorch1.5
    ```

### Getting Started

Please refer to [GETTING_STARGTED.md](docs/GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints.

### Model Zoo

The models with ResNet-50 backbone and ResNet-101 backbone are available in [[Gdrive]](https://drive.google.com/drive/folders/17CVnc5XOyqqDlg1veXRE9hY9r123Nvqx?usp=sharing)

<table border="2">
    <thead>
        <tr>
            <th colspan=1> </th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO </th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO+</th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCOg</th>
            <th colspan=2> ReferItGame</th>
        </tr>
    </thead>
    <tbody>
    <tr>    
            <td> </td>
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>g-val</td>
            <td>u-val</td>
            <td>u-test</td>
            <td>val</td>
            <td>test</td>
        </tr>
    </tbody>
    <tbody>
    <tr>
            <td> R-50 </td>
            <td>80.5</td>
            <td>83.2</td>
            <td>75.2</td>
            <td>66.4</td>
            <td>70.5</td>
            <td>57.7</td>
            <td>66.4</td>
            <td>67.9</td>
            <td>67.4</td>
            <td>71.6</td>
            <td>69.3</td>
        </tr>
    </tbody>
    <tbody>
    <tr>
            <td> R-101 </td>
            <td>80.8</td>
            <td>83.4</td>
            <td>76.9</td>
            <td> 68.0 </td>
            <td> 72.5</td>
            <td> 59.2</td>
            <td> 68.0 </td>
            <td>68.7</td>
            <td>68.0</td>
            <td> - </td>
            <td> - </td>
        </tr>
    </tbody>
</table>


### Training and Evaluation

1.  Training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50 --epochs 90 --lr_drop 60
    ```

    We recommend to set --max_query_len 40 for RefCOCOg, and --max_query_len 20 for other datasets. 
    
    We recommend to set --epochs 180 (--lr_drop 120 acoordingly) for RefCOCO+, and --epochs 90 (--lr_drop 60 acoordingly) for other datasets. 

2.  Evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50
    ```

### Acknowledge
This codebase is partially based on [ReSC](https://github.com/zyang-ur/ReSC) and [DETR](https://github.com/facebookresearch/detr).
