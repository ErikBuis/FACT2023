import json
import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import InterpolationMode

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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


class VisualGenome(Dataset):
    """Visual Genome dataset."""

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
        Get a single instance from the dataset. An "instance" refers to a
        specific occurrence of a certain object in a certain image.

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

        # Get the image that contains the instance and apply augmentations.
        path = f"VG_100K/{self.samples[sample_id]['image_id']}.jpg"
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Create a one-hot vector for the target category.
        target = torch.zeros((len(self.labels),))
        target[self.labels[label]] = 1

        # Create the instance mask.
        bbox = self.samples[sample_id]["objects"][label][bbox_id]
        mask = torch.zeros(img.size)
        mask[bbox["y"]:bbox["y"] + bbox["h"],
             bbox["x"]:bbox["x"] + bbox["w"]] = 1
        mask = self.mask_transform(mask)

        return img, target, mask

    def _load_samples(self) -> list[dict[str, list[dict[str, int]]]]:
        """Load the object data from the json file."""
        print("Loading object data...")
        path = os.path.join(self.root, "vg_objects_preprocessed.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. "
                                    "Please run `preprocess.py` first.")
        with open(path) as f:
            samples = json.load(f)
        print("Loading complete.")
        return samples

    def _load_labels(self) -> dict[str, int]:
        """Load the labels from the pickle file."""
        print("Loading labels...")
        path = os.path.join(self.root, "vg_labels.pkl")
        with open(path, "rb") as f:
            labels = pickle.load(f)
        print("Loading complete.")
        return labels


class MyCocoDetection(CocoDetection):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Image in which the instance is found.
                - Multiple-hot target category vector.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        target_list = []
        for t in target:
            mask = coco.annToMask(t)
            im = Image.fromarray(mask)
            mask = np.array(im.resize((224, 224)))
            obj = list(label_embedding["stoi"].keys())[t["category_id"] - 1]
            idxs = list(label_embedding["stoi"].values())[t["category_id"] - 1]
            mask_dict = {"mask": mask, "object": obj, "idx": idxs}
            target_list.append(mask_dict)

        path = coco.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target_list


class MyCocoSegmentation(CocoDetection):
    """Coco dataset with segmentation masks."""

    def __init__(self, root: str, annFile: str,
                 transform: Optional[Callable],
                 mask_width: int = 7,
                 mask_height: int = 7,
                 inference: bool = False):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            root (str): Root directory where Coco images were downloaded to.
            annFile (str): Path to json file containing instance annotations.
            transform (Optional[Callable]): Transform to apply to the images.
            mask_width (int, optional): Width that the segmentation mask should
                be resized to. Defaults to 7.
            mask_height (int, optional): Height that the segmentation mask
                should be resized to. Defaults to 7.
            inference (bool, optional): If True, the dataset will return the
                image id instead of the image (?). Defaults to False.
        """
        super(CocoDetection, self).__init__(root, transform=transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.anns.keys())
        self.mask_transform = create_mask_transform(mask_width, mask_height)
        self.inference = inference

        # Load the object embedding vectors. `label_embedding` is a dictionary
        # that contains the following entries:
        # - "stoi" (String TO Index): a dictionary that maps a category name to
        #   its indices in the embedding matrix (can be multiple).
        # - "itos" (Index TO String): a dictionary that maps an index in the
        #   embedding matrix to its category name.
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        self.label_embedding = torch.load(label_embedding_file)

        # - "idtov" (ID TO Vector): a dictionary that maps a category id
        #   to its multiple-hot vector.
        empty_vector = torch.zeros((len(self.label_embedding["itos"]),))
        self.label_embedding["idtov"] = {}
        for category_id, indices in \
                enumerate(self.label_embedding["stoi"].values(), start=1):
            self.label_embedding["idtov"][category_id] = empty_vector.clone()
            for index in indices:
                idx_in_vector = list(self.label_embedding["itos"]).index(index)
                self.label_embedding["idtov"][category_id][idx_in_vector] = 1

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset. An "instance" refers to a
        specific occurrence of a certain object in a certain image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image in which the instance is found.
                    Shape: [3, 224, 224]
                - Multiple-hot target category vector.
                    Shape: [num_categories]
                - Instance mask.
                    Shape: [1, mask_width, mask_height]
        """
        # Use a separate value to prevent frequent lookups.
        coco = self.coco
        label_embedding = self.label_embedding

        # Get the instance annotation from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "segmentation": a list of polygons that define the instance's
        #   boundaries. Each polygon is a list of points (x, y).
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        ann = coco.anns[self.ids[index]]

        # TODO Look at whether this "improved code" is correct (second author).
        # Added an improved version.
        if self.inference:
            ann_ids = coco.getAnnIds(imgIds=ann["image_id"])
            target = coco.loadAnns(ann_ids)
            target_list = []
            for t in target:
                mask = coco.annToMask(t)
                im = Image.fromarray(mask)
                mask = np.array(im.resize((224, 224)))
                obj, idxs = \
                    list(label_embedding["stoi"].items())[t["category_id"] - 1]
                mask_dict = {"mask": mask, "object": obj, "idx": idxs}
                target_list.append(mask_dict)

        # Get the image that contains the instance and apply augmentations.
        path = coco.loadImgs(ann["image_id"])[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Create a multiple-hot vector for the target category.
        target = label_embedding["idtov"][ann["category_id"]]

        # Create the instance mask.
        mask = coco.annToMask(ann)
        mask = self.mask_transform(mask)

        # TODO Look at whether this "improved code" is correct (second author).
        if self.inference:
            return img, target_list

        return img, target, mask
