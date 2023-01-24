import json
import os
import pickle
from typing import Callable, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchtext.vocab import GloVe
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

    def __init__(self,
                 root: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        self.root = root
        self.transform = transform
        self.samples = self._load_samples()
        self.labels = self._load_labels()
        self.mask_transform = create_mask_transform(mask_width, mask_height)

    def _load_samples(self) -> list[dict[str, list[dict[str, int]]]]:
        """Load the object data from the json file."""
        print("Loading object data...", end=" ")
        path = os.path.join(self.root, "vg_objects_preprocessed.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. "
                                    "Please run `preprocess.py` first.")
        with open(path) as f:
            samples = json.load(f)
        print("Done.")
        return samples

    def _load_labels(self) -> dict[str, int]:
        """Load the labels from the pickle file."""
        print("Loading labels...", end=" ")
        path = os.path.join(self.root, "vg_labels.pkl")
        with open(path, "rb") as f:
            labels = pickle.load(f)
        print("Done.")
        return labels


class VisualGenomeImages(_VisualGenomeAbstract):
    """Visual Genome dataset that returns images."""
    
    def __init__(self,
                 root: str,
                 transform: Optional[Callable],
                 glove: GloVe,
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            transform (Optional[Callable]): Transform to apply to the images.
            glove (Glove): GloVe word embeddings.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, transform, mask_width, mask_height)
        self.glove = glove
        
        # Remove samples that don't have any labels in self.labels.
        remove_indices = []
        for sample_id, sample in enumerate(self.samples):
            if not any(label in self.labels 
                       for label in sample["objects"].keys()):
                remove_indices.append(sample_id)
        self.samples = [sample for i, sample in enumerate(self.samples)
                        if i not in remove_indices]

    def __len__(self) -> int:
        """Get the amount of images in the VG dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image from the dataset.
                    Shape: [3, 224, 224]
                - One-hot target category vector for each instance.
                    Shape: [num_instances]
                - Instance mask for each instance.
                    Shape: [num_instances, 1, mask_width, mask_height]
        """
        sample_id = idx

        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[sample_id]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        targets = []
        masks = []
        for label, bboxes in self.samples[sample_id]["objects"].items():
            if label not in self.labels:
                continue
            for bbox in bboxes:
                # Get the glove index that corresponds to this label.
                targets.append(self.glove.stoi[label])

                # Load the segmentation mask for each instance.
                mask = torch.zeros(img_og.size)
                mask[bbox["y"]:bbox["y"] + bbox["h"],
                     bbox["x"]:bbox["x"] + bbox["w"]] = 1
                mask = self.mask_transform(mask)
                masks.append(mask)

        targets = torch.stack(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class VisualGenomeInstances(_VisualGenomeAbstract):
    """Visual Genome dataset that returns instances."""

    def __init__(self,
                 root: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            root (str): The root directory where VG images were downloaded to.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, transform, mask_width, mask_height)

        # Create a mapping from instances to bboxes.
        self._instance_to_bbox = []
        for sample_id, sample in enumerate(self.samples):
            for label, bboxes in sample["objects"].items():
                if label not in self.labels:
                    continue
                for bbox_id in range(len(bboxes)):
                    self._instance_to_bbox.append((sample_id, label, bbox_id))

    def __len__(self) -> int:
        """Get the amount of instances in the VG dataset."""
        return len(self._instance_to_bbox)

    def __getitem__(self, idx: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
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
        sample_id, label, bbox_id = self._instance_to_bbox[idx]

        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[sample_id]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create a one-hot target vector for the instance.
        target = torch.zeros((len(self.labels),))
        target[self.labels[label]] = 1

        # Load the segmentation mask for the instance.
        bbox = self.samples[sample_id]["objects"][label][bbox_id]
        mask = torch.zeros(img_og.size)
        mask[bbox["y"]:bbox["y"] + bbox["h"],
             bbox["x"]:bbox["x"] + bbox["w"]] = 1
        mask = self.mask_transform(mask)

        return img, target, mask


class _CocoAbstract(Dataset):
    """Abstract class for the COCO dataset."""

    def __init__(self, root: str, annFile: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where Coco images were downloaded to.
            annFile (str): Path to json file containing instance annotations.
            cat_mappings_file (str): Path to pickle file containing categories.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__()
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
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
    """Coco dataset that returns images."""
    
    def __init__(self, root: str, annFile: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where Coco images were downloaded to.
            annFile (str): Path to json file containing instance annotations.
            cat_mappings_file (str): Path to pickle file containing categories.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, annFile, cat_mappings_file, transform, 
                         mask_width, mask_height)

        # Remove samples that don't have any annotations.
        remove_ids = []
        for index in range(len(self.ids)):
            anns = self.coco.loadAnns(self.coco.getAnnIds(self.ids[index]))
            if len(anns) == 0:
                remove_ids.append(index)
        self.ids = [self.ids[i] for i in range(len(self.ids)) 
                    if i not in remove_ids]

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple[Image.Image, torch.Tensor, torch.Tensor]:
                - Image from the dataset.
                    Shape: [3, 224, 224].
                - Multiple-hot target category vector for each instance.
                    Shape: [num_instances].
                - Instance mask for each instance.
                    Shape: [num_instances, 1, mask_width, mask_height].
        """
        # Get the instance annotations from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        anns = self.coco.loadAnns(self.coco.getAnnIds(self.ids[index]))
        
        # Get the image and apply augmentations.
        path = self.coco.loadImgs(self.ids[index])[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Create the target category vector and instance masks.
        targets = []
        masks = []
        for ann in anns:
            mask = self.mask_transform(self.coco.annToMask(ann))
            tokens = self.coco.cats[ann["category_id"]]["name"].split()
            glove_indices = [self.cat_mappings["stoi"][token] 
                             for token in tokens]
            for glove_idx in glove_indices:
                # Add the GloVe index to the list of targets.
                targets.append(glove_idx)
                # Add the mask to the list of masks.
                masks.append(mask)
        targets = torch.tensor(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class CocoInstances(_CocoAbstract):
    """Coco dataset that returns instances."""

    def __init__(self, root: str, annFile: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where Coco images were downloaded to.
            annFile (str): Path to json file containing instance annotations.
            cat_mappings_
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
        """
        super().__init__(root, annFile, cat_mappings_file, transform,
                         mask_width, mask_height)
        self.ids = sorted(self.coco.anns.keys())

        # Define a dict that maps a GloVe index to the index in the 
        # multiple-hot vector that we construct in __getitem__.
        self.glove_idx_to_idx_in_vector = {}
        for i, glove_idx in enumerate(self.cat_mappings["itos"].keys()):
            self.glove_idx_to_idx_in_vector[glove_idx] = i

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple[Image.Image, torch.Tensor, torch.Tensor]:
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
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Create a multiple-hot target vector for the instance.
        target = torch.zeros((len(self.cat_mappings["itos"]),))
        tokens = self.coco.cats[ann["category_id"]]["name"].split()
        for token in tokens:
            glove_idx = self.cat_mappings["stoi"][token]
            idx_in_vector = self.glove_idx_to_idx_in_vector[glove_idx]
            target[idx_in_vector] = 1

        # Load the segmentation mask for the instance.
        mask = self.mask_transform(self.coco.annToMask(ann))

        return img, target, mask
