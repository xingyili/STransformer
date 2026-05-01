# STransformer

## description

![STransformer](./STransformer.png)

## Getting Started

### dependencies

* Python 3.10.15
* Pytorch 1.12.1
* numpy 1.24.4
* pandas 1.4.4
* scanpy 1.9.8
* scikit-learn 1.3.2
* matplotlib 3.7.5

### Data organization
Before running STransformer, please organize the input data as follows. Taking the DLPFC slice 151507 as an example:

```text
data/
├── DLPFC/
│   └── 151507/
│       ├── spatial/
│       │   ├── full_image.tif
│       │   ├── scalefactors_json.json
│       │   ├── tissue_hires_image.png
│       │   ├── tissue_lowres_image.png
│       │   └── tissue_positions_list.csv       
│       ├── 151507_truth.txt
│       ├── filtered_feature_bc_matrix.h5
│       ├── metadata.tsv
│       └── position_tensor.pt
└── image_feature/
    └── DLPFC/
        └── 151507/
            └── embeddings.npy
```
Users can also modify `load_data.py` to flexibly adapt STransformer to custom data organizations.


### How to run

If you want to manually setup STransformer, we recommend you to use [Anaconda](https://docs.anaconda.com/free/anaconda/install/) to build the runtime environment.

Step 1: Clone this repository from Github:

```
git clone https://github.com/xingyili/STransformer.git
cd STransformer
```

Step 2: Extract image features by BYOL (replace with your own data path):

```
cd ./BYOL
python image_extraction.py
```

Step 3: Run STransformer (example: DLPFC 151507 slice):

```
cd ..
python main.py --cuda 0 --dataset DLPFC --slice_name 151507 --t_epoch 500
```

### Key parameters
The main parameters used to run STransformer include:
- `--dataset`: dataset name.
- `--slice_name`: slice ID.
- `--t_epoch`: number of training epochs.
- `--lr`: learning rate.
- `--drop_feature_rate`: feature masking rate during training.
- `--drop_edge_rate`: edge dropping rate during training.
- `--cuda`: GPU device ID.

