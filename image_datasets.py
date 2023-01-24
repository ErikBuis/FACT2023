import json
import os
import pickle
from typing import Callable, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

# Define the ImageNet normalization parameters.
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# Define the transforms for the different datasets.
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
}


def create_mask_transform(mask_width: int, mask_height: int) \
        -> transforms.Compose:
    """
    Create a transform to resize a mask.

    Args:
        mask_width (int): Width of the mask.
        mask_height (int): Height of the mask.

    Returns:
        transforms.Compose: Mask transform.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        transforms.Resize([mask_height, mask_width],
                          interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


class _VisualGenomeAbstract(Dataset):
    """Abstract class for the Visual Genome dataset."""

    def __init__(self, root: str, objs_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            objs_file (str): Path to the json file containing the preprocessed
                object data.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        self.root = root
        self.transform = transform
        self.mask_transform = create_mask_transform(mask_width, mask_height)
        with open(objs_file) as f:
            self.samples = json.load(f)

        # Load the categories. `self.cat_mappings` is a dictionary that
        # contains the following entries:
        # - "stoi" (String TO Index): a dictionary that maps a VG category
        #   token to its GloVe index in the embedding matrix.
        # - "itos" (Index TO String): a dictionary that maps a GloVe index in
        #   the embedding matrix to its VG category token.
        with open(cat_mappings_file, "rb") as f:
            self.cat_mappings = pickle.load(f)


class VisualGenomeImages(_VisualGenomeAbstract):
    """Visual Genome dataset that returns images."""

    def __init__(self, root: str, objs_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            objs_file (str): Path to the json file containing the preprocessed
                object data.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, objs_file, cat_mappings_file,
                         transform, mask_width, mask_height)

        # Remove samples that don't have any categories.
        self.samples = [sample for sample in self.samples
                        if all(cat_token in self.cat_mappings["stoi"].keys()
                               for cat_token in sample["objects"].keys())]

    def __len__(self) -> int:
        """Get the amount of images in the VG dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image from the dataset.
                    Shape: [3, 224, 224]
                - GloVe category index for each instance.
                    Shape: [num_instances]
                - Mask for each instance.
                    Shape: [num_instances, 1, mask_width, mask_height]
        """
        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[index]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create the target category vector and instance masks.
        targets = []
        masks = []
        for cat_token, bboxes in self.samples[index]["objects"].items():
            if cat_token not in self.cat_mappings["stoi"].keys():
                continue
            for bbox in bboxes:
                # Add the GloVe index to the list of targets.
                targets.append(self.cat_mappings["stoi"][cat_token])
                # Add the mask to the list of masks.
                mask = torch.zeros(img.shape)
                mask[bbox["y"]:bbox["y"] + bbox["h"],
                     bbox["x"]:bbox["x"] + bbox["w"]] = 1
                masks.append(self.mask_transform(mask))
        targets = torch.tensor(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class VisualGenomeInstances(_VisualGenomeAbstract):
    """Visual Genome dataset that returns instances."""

    def __init__(self, root: str, objs_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            objs_file (str): Path to the json file containing the preprocessed
                object data.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, objs_file, cat_mappings_file,
                         transform, mask_width, mask_height)

        # Create a mapping from instances to bboxes.
        self.instance2idx = []
        for sample_idx, sample in enumerate(self.samples):
            for cat_token, bboxes in sample["objects"].items():
                if cat_token not in self.cat_mappings["stoi"].keys():
                    continue
                for bbox_idx in range(len(bboxes)):
                    self.instance2idx.append((sample_idx, cat_token, bbox_idx))

        # Define a dict that maps a GloVe index to its index in the
        # multiple-hot target vectors.
        self.glove2vec_idx = {}
        for i, glove_idx in enumerate(self.cat_mappings["stoi"].values()):
            self.glove2vec_idx[glove_idx] = i

    def __len__(self) -> int:
        """Get the amount of instances in the VG dataset."""
        return len(self.instance2idx)

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image in which the instance is found.
                    Shape: [3, 224, 224]
                - One-hot target category vector.
                    Shape: [num_categories]
                - Instance mask.
                    Shape: [1, mask_width, mask_height]
        """
        sample_idx, cat_token, bbox_idx = self.instance2idx[index]

        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[sample_idx]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create a one-hot target vector for the instance.
        target = torch.zeros((len(self.cat_mappings["stoi"]),))
        glove_idx = self.cat_mappings["stoi"][cat_token]
        target[self.glove2vec_idx[glove_idx]] = 1

        # Load the segmentation mask for the instance.
        bbox = self.samples[sample_idx]["objects"][cat_token][bbox_idx]
        mask = torch.zeros(img.shape)
        mask[bbox["y"]:bbox["y"] + bbox["h"],
             bbox["x"]:bbox["x"] + bbox["w"]] = 1
        mask = self.mask_transform(mask)

        return img, target, mask


class _CocoAbstract(Dataset):
    """Abstract class for the COCO dataset."""

    def __init__(self, root: str, ann_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where COCO images were downloaded to.
            ann_file (str): Path to json file containing instance annotations.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__()
        self.root = root
        self.coco = COCO(ann_file)
        self.transform = transform
        self.ids = sorted(self.coco.imgs.keys())
        self.mask_transform = create_mask_transform(mask_width, mask_height)
        self.mask_width = mask_width
        self.mask_height = mask_height

        # Load the categories. `self.cat_mappings` is a dictionary that
        # contains the following entries:
        # - "stoi" (String TO Index): a dictionary that maps a COCO category
        #   token to its GloVe index in the embedding matrix.
        # - "itos" (Index TO String): a dictionary that maps a GloVe index in
        #   the embedding matrix to its COCO category token.
        with open(cat_mappings_file, "rb") as f:
            self.cat_mappings = pickle.load(f)

    def __len__(self) -> int:
        """Get the amount of instances/images in the COCO dataset."""
        return len(self.ids)


class CocoImages(_CocoAbstract):
    """COCO dataset that returns images."""

    def __init__(self, root: str, ann_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where COCO images were downloaded to.
            ann_file (str): Path to json file containing instance annotations.
            cat_mappings_file (str): Path to pickle file containing categories.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, ann_file, cat_mappings_file,
                         transform, mask_width, mask_height)

        # Remove samples that don't have any annotations.
        self.ids = [image_id for image_id in self.ids
                    if len(self.coco.getAnnIds(image_id)) > 0]

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image from the dataset.
                    Shape: [3, 224, 224].
                - GloVe category index for each instance.
                    Shape: [num_instances].
                - Mask for each instance.
                    Shape: [num_instances, 1, mask_width, mask_height].
        """
        # Get the instance annotations from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        anns = self.coco.loadAnns(self.coco.getAnnIds(self.ids[index]))

        # Get the image and apply augmentations.
        path = self.coco.loadImgs(self.ids[index])[0]["file_name"]
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create the target category vector and instance masks.
        targets = []
        masks = []
        for ann in anns:
            mask = self.mask_transform(self.coco.annToMask(ann))
            cat_tokens = self.coco.cats[ann["category_id"]]["name"].split()
            glove_indices = [self.cat_mappings["stoi"][cat_token]
                             for cat_token in cat_tokens]
            for glove_idx in glove_indices:
                # Add the GloVe index to the list of targets.
                targets.append(glove_idx)
                # Add the mask to the list of masks.
                masks.append(mask)
        targets = torch.tensor(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class CocoInstances(_CocoAbstract):
    """COCO dataset that returns instances."""

    def __init__(self, root: str, ann_file: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where COCO images were downloaded to.
            ann_file (str): Path to json file containing instance annotations.
            cat_mappings_file (str): Path to pickle file containing categories.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, ann_file, cat_mappings_file,
                         transform, mask_width, mask_height)
        self.ids = sorted(self.coco.anns.keys())

        # Define a dict that maps a GloVe index to its index in the
        # multiple-hot target vectors.
        self.glove2vec_idx = {}
        for i, glove_idx in enumerate(self.cat_mappings["stoi"].values()):
            self.glove2vec_idx[glove_idx] = i

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image in which the instance is found.
                    Shape: [3, 224, 224].
                - Multiple-hot target category vector.
                    Shape: [num_categories].
                - Instance mask.
                    Shape: [1, mask_width, mask_height].
        """
        # Get the instance annotation from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        ann = self.coco.loadAnns(self.ids[index])[0]

        # Get the image and apply augmentations.
        path = self.coco.loadImgs(ann["image_id"])[0]["file_name"]
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create a multiple-hot target vector for the instance.
        target = torch.zeros((len(self.cat_mappings["stoi"]),))
        cat_tokens = self.coco.cats[ann["category_id"]]["name"].split()
        for cat_token in cat_tokens:
            glove_idx = self.cat_mappings["stoi"][cat_token]
            target[self.glove2vec_idx[glove_idx]] = 1

        # Load the segmentation mask for the instance.
        mask = self.mask_transform(self.coco.annToMask(ann))

        return img, target, mask
