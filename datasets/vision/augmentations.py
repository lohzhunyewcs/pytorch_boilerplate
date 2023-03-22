import albumentations as A
# import albumentations.pytorch as transforms
from albumentations.pytorch import ToTensorV2
import torch
import random
from torchvision import transforms

# DEFAULT_DATA_TRANSFORMS = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
# DEFAULT_DATA_TRANSFORMS = {
#     'train': A.Compose([
#         A.Resize((256, 256)),
#         A.RandomResizedCrop(224),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# DEFAULT_DATA_TRANSFORMS['test'] = DEFAULT_DATA_TRANSFORMS['val']

def create_transforms(
        is_train: bool, img_width: int, img_height: int,
        use_random_crop: bool=False,
        use_weathers: bool=False,
        use_simple_only: bool=True
    ) -> torch.TensorType:
    transforms_list = []
    

    # Resize
    transforms_list.append(
        A.Resize(img_height, img_width)
    )

    if use_random_crop:
        transforms_list.append(A.RandomResizedCrop(img_height, img_width))

    some_of_list = []

    some_of_list.append(A.HorizontalFlip())
    some_of_list.append(A.VerticalFlip())
    some_of_list.append(A.RandomBrightnessContrast(p=0.2))
    some_of_list.append(A.ShiftScaleRotate())
    some_of_list.append(A.RGBShift())
    # some_of_list.append(A.Blur())
    some_of_list.append(A.GaussNoise())
    some_of_list.append(A.ElasticTransform())
    some_of_list.append(A.Cutout(p=0.1))

    some_of_list.append(A.augmentations.geometric.rotate.Rotate(limit=180)) 

    if not use_simple_only:

        some_of_list += [
            # A.CLAHE(),
            # A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
        ]
    

    weather_list = []
    weather_list.append(A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=0.5))
    weather_list.append(A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5))
    weather_list.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1))
    weather_list.append(A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=0.5))
    weather_list.append(A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1))
    weather_list.append(A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1))
    weather_transforms = A.OneOf(
        A.Compose(weather_list)
    )

    # For masks
    # some_of_list.append(A.MaskDropout((10,15), p=0.5))

    if is_train:
        transforms_list.append(
            A.SomeOf(
                A.Compose(some_of_list),
                n=random.randint(0, len(some_of_list) //2)
            )
        )
        # print(f'some of {transforms_list = }')
        if use_weathers:
            transforms_list.append(
                weather_transforms
            )
        # print(f'some of {weather_transforms = }')

    transforms_list += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)

def use_torch_augmentations(is_train:bool):
    if is_train:
        img_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        img_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return img_transforms