from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union
from torchvision import transforms
import cv2
import torch
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image

DEFAULT_DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class ImageClassificationDataset(Dataset):
    def __init__(
        self, image_paths: List[str], gts: List[int],
        gt_to_cat_name: dict[int, str],
        img_transforms: Union[BasicTransform, transforms.transforms.Compose],
        label_transforms: BasicTransform
    ) -> None:
        super().__init__()
        assert len(image_paths) == len(gts)
        self.image_paths = image_paths
        self.gts = gts
        self.gt_to_cat_name = gt_to_cat_name
        self.num_class = len(gt_to_cat_name)
        self.img_transforms = img_transforms
        self.label_transforms = label_transforms
        self.debug_mode = False

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        # img = cv2.imread(img_path)
        # img = Image.open(img_path)

        # print(f'before transform')
        # print(f'{type(img) = }, {img.shape = }')
        assert self.img_transforms is not None
        if type(self.img_transforms) == transforms.transforms.Compose:
            # img = torch.from_numpy(img)
            img = Image.open(img_path)
            img = self.img_transforms(img)
        else:
            img = cv2.imread(img_path)
            img = self.img_transforms(image=img)['image']

        # print(f'after transform')
        # print(f'{type(img) = }')

        # print(f'Generating GT')
        gt = self.gts[idx]
        # gt_list = [0] * self.num_class
        # if self.num_class == 1:
        #     gt_list[0] = gt
        # else:
        #     gt_list[gt] = 1
        # gt_tensor = torch.tensor(gt_list, dtype=torch.float)

        # gt_list = [0] * self.num_class
        # gt_list[gt] = 1

        # gt_tensor = torch.tensor(gt_list).float()
        # print(f'Done Generating GT')
        gt_tensor = torch.tensor(gt)

        
        if self.debug_mode:
            return img_path, gt
        else:
            return img, gt_tensor
        
def get_dataset_and_dataloader(
        image_paths: List[str], gts: List[int],
        gt_to_cat_name: dict[int, str],
        img_transforms: transforms,
        label_transforms: transforms,
        batch_size: int,
        shuffle: bool
    ):
    dataset = ImageClassificationDataset(
            image_paths, gts, gt_to_cat_name, img_transforms, label_transforms
    )
    datasize = len(dataset)

    dataloader = DataLoader(
        dataset, batch_size, shuffle
    )
    return {
        'dataset': dataset,
        'datasize': datasize,
        'dataloader': dataloader
    }

