import argparse
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from byol_pytorch import BYOL
from torchvision import models
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import random
from torch import nn
import time
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from adata_processing import LoadSingle10xAdata


def clip_to_patches(data_dir, dataset, slice_name):
    path = data_dir
    label=True
    d=144
    if dataset in ['DLPFC', 'chicken_heart']:
        patch_size = int(3.5*d)
    elif dataset == 'AD':
        patch_size = 80

    # Initialize data loader
    if dataset == 'DLPFC':
        loader = LoadSingle10xAdata(path=path, image_emb=False, label=label, filter_na=True, slice_name=slice_name)
        loader.load_data()
        if label:
            loader.load_label()
        adata = loader.adata
    elif dataset == 'AD':
        data_path = f'../data/AD/{slice_name}/{slice_name}.h5ad'
        adata = sc.read(data_path)
    elif dataset == 'chicken_heart':
        data_path = f'../data/chicken_heart/{slice_name}/chicken_heart_{slice_name}.h5ad'
        adata = sc.read(data_path)

    if dataset == 'DLPFC':
        if os.path.exists(os.path.join(path, "spatial", "full_image.tif")):
            print("File exists.")
        else:
            print("File does not exist.")
    elif dataset == 'AD':
        if os.path.exists(os.path.join(path, "spatial", "tissue_hires_image.png")):
            print("File exists.")
        else:
            print("File does not exist.")
    elif dataset == 'chicken_heart':
        if os.path.exists(os.path.join(path, f"chicken_heart_spatial_RNAseq_{slice_name}_image.tif")):
            print("File exists.")
        else:
            print("File does not exist.")

    # Read image
    if dataset == 'DLPFC':
        im = cv2.imread(os.path.join(path, "spatial", "full_image.tif"), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("Failed to load the image. Check the file path.")
    elif dataset == 'AD':
        im = cv2.imread(os.path.join(path, "spatial", "tissue_hires_image.png"), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("Failed to load the image. Check the file path.")
    elif dataset == 'chicken_heart':
        im = cv2.imread(os.path.join(path, f"chicken_heart_spatial_RNAseq_{slice_name}_image.tif"), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("Failed to load the image. Check the file path.")

    # Create directory for clipped images
    clip_image_path = os.path.join(path, 'clip_image')
    try:
        os.makedirs(clip_image_path)
        print("Folder 'clip_image' created successfully")
    except FileExistsError:
        print("Folder 'clip_image' already exist")

    # Process and save patches
    patches = []
    for i, coord in tqdm(enumerate(adata.obsm['spatial']), total=len(adata.obsm['spatial'])):
        # Calculate patch coordinates
        left = int(coord[0] - patch_size / 2)
        top = int(coord[1] - patch_size / 2)
        right = left + patch_size
        bottom = top + patch_size

        # Extract patch
        patch = im[top:bottom, left:right]

        # Resize patch to 512x512, using INTER_LINEAR for both upsizing and downsizing
        if patch_size != 512:
            resized_patch = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR)
        else:
            resized_patch = patch
        # Save resized patch
        cv2.imwrite(os.path.join(clip_image_path, f'{i}.png'), resized_patch)


def process_image(filename, path, output_path, GaussianBlur, lower, upper):
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, 0)

    # Initial brightness increase
    # image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    original_height, original_width = image.shape[:2]

    resize_needed = (original_height != 512 or original_width != 512)
    if resize_needed:
        image = cv2.resize(image, (512, 512))

    if GaussianBlur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    image_shape = image.shape
    custom_mask = create_custom_mask(image_shape, lower, lower, upper, upper)
    fshift_masked = fshift * custom_mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    image_filtered = np.fft.ifft2(f_ishift)
    image_filtered = np.abs(image_filtered)

    # Post-processing brightening
    # image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # image_filtered = np.clip(image_filtered * 1.2, 0, 255).astype(np.uint8)

    if GaussianBlur:
        image_filtered = cv2.GaussianBlur(image_filtered, (15, 15), 0)
    image_filtered_rgb = cv2.cvtColor(np.float32(image_filtered), cv2.COLOR_GRAY2RGB)
    mae_patch = cv2.resize(image_filtered_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    if resize_needed:
        image_filtered_rgb = cv2.resize(image_filtered_rgb, (original_width, original_height))
    cv2.imwrite(os.path.join(output_path, filename), image_filtered_rgb)
    return mae_patch

def create_custom_mask(image_shape, x1, y1, x2, y2):
    rows, cols = image_shape
    mask = np.zeros((rows, cols), np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask

def process_images(dataset_name, slice_name, epoch_num):
    
    if dataset_name == 'DLPFC':
        path = f"../data/DLPFC/{slice_name}/clip_image_filter"
    elif dataset_name == 'AD':
        path = f"../data/AD/{slice_name}/clip_image_filter"
    elif dataset_name == 'chicken_heart':
        path = f"../data/chicken_heart/{slice_name}/clip_image_filter"

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    torch.use_deterministic_algorithms(True)

    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_list = sorted(
                [x for x in os.listdir(root_dir) if x.endswith('.png')],
                key=lambda x: int(x.split('.')[0])
                )


        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.image_list[idx])
            img = Image.open(img_name).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=41, shuffle=True, num_workers=0) 

    class RandomApply(nn.Module):
        def __init__(self, fn, p):
            super().__init__()
            self.fn = fn
            self.p = p

        def forward(self, x):
            if random.random() > self.p:
                return x
            return self.fn(x)

    DEFAULT_AUG = torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomResizedCrop((256, 256)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64),
            std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)),

    )

    learner = BYOL(
            models.resnet50(pretrained=True),
            image_size=256,
            hidden_layer='avgpool',
            augment_fn=DEFAULT_AUG
        ).double()
    if torch.cuda.is_available():
        learner = learner.cuda()

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    print('start training!')
    for epoch in range(epoch_num):
        start_time = time.time()
        for images in tqdm(data_loader, desc=f"Processing images", leave=False):
            images = images.cuda().double() if torch.cuda.is_available() else images.double()
            loss = learner(images)
            # print(f'Loss: {loss.item():.8f}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        end_time = time.time()
        # print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')

    torch.save(learner.state_dict(), 'learner.pth')

    learner.eval()
    embeddings = []
    print('start eval!')
    for i in tqdm(range(len(dataset)), desc='Evaluation Progress'):
        img = dataset[i]
        img = img.cuda().double() if torch.cuda.is_available() else img.double()
        with torch.no_grad():
            _, embedding = learner(img.unsqueeze(0), return_embedding=True)
            embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)
    np.save(f'../image_feature/{dataset_name}/{slice_name}/embeddings.npy', embeddings)
    print(embeddings.shape)

if __name__ == '__main__':
    dataset = 'DLPFC'
    slice_name = '151507'
    
    data_dir = f'../data/{dataset}/{slice_name}'

    ### step 1:clip to patches ###
    clip_to_patches(data_dir, dataset, slice_name)
    
    ### step 2:Filter patches ###
    path = os.path.join(data_dir,'clip_image')
    output_path =  os.path.join(data_dir,'clip_image_filter')
    GaussianBlur = True
    upper = 275
    lower = 245
    try:
        os.makedirs(output_path)
        print("Folder 'clip_image_filter' created successfully")
    except FileExistsError:
        print("Folder 'clip_image_filter' already exist")
    png_files = [name for name in os.listdir(path) if name.endswith('.png')]
    args = [(filename, path, output_path, GaussianBlur, lower, upper) for filename in png_files]

    with Pool(processes=os.cpu_count()) as pool:
        patches = pool.starmap(process_image, args)

    ### step 3:BYOL###
    epoch_num=1
    process_images(dataset, slice_name, epoch_num)
