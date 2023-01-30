import os

from image_datasets import CocoImages, CocoInstances, data_transforms

root = "./data/coco"

# Count the number of images in the training dataset.
coco_images_train = CocoImages(
    ann_file=os.path.join(root, "annotations/instances_train2017.json"),
    root=os.path.join(root, "train2017"),
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=data_transforms["train"],
    filter_width=224,
    filter_height=224
)
amount_images_train = len(coco_images_train)
print(f"Amount of images in the training set: {amount_images_train}")

# Count the number of images in the validation dataset.
coco_images_val = CocoImages(
    ann_file=os.path.join(root, "annotations/instances_val2017.json"),
    root=os.path.join(root, "val2017"),
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=data_transforms["val"],
    filter_width=224,
    filter_height=224
)
amount_images_val = len(coco_images_val)
print(f"Amount of images in the validation set: {amount_images_val}")

# Count the number of instances in the training dataset.
coco_instances_train = CocoInstances(
    ann_file=os.path.join(root, "annotations/instances_train2017.json"),
    root=os.path.join(root, "train2017"),
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=data_transforms["train"],
    filter_width=224,
    filter_height=224
)
amount_instances_train = len(coco_instances_train)
print(f"Amount of instances in the training set: {amount_instances_train}")

# Count the number of instances in the validation dataset.
coco_instances_val = CocoInstances(
    ann_file=os.path.join(root, "annotations/instances_val2017.json"),
    root=os.path.join(root, "val2017"),
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=data_transforms["val"],
    filter_width=224,
    filter_height=224
)
amount_instances_val = len(coco_instances_val)
print(f"Amount of instances in the validation set: {amount_instances_val}")

# Count how often each category occurs in the training dataset.
cat_mappings = coco_images_train.cat_mappings
coco = coco_images_train.coco
category_counts_train = {}
for img_id in coco_images_train.img_ids:
    anns = coco.loadAnns(coco.getAnnIds(img_id))
    for ann in anns:
        cat = coco.loadCats(ann["category_id"])[0]
        for cat_token in cat["name"].split():
            if cat_token in cat_mappings["stoi"]:
                if cat_token not in category_counts_train:
                    category_counts_train[cat_token] = 0
                category_counts_train[cat_token] += 1
amount_categories_train = len(category_counts_train)
cats_sorted_train = sorted(category_counts_train.items(), key=lambda x: x[1],
                           reverse=True)
print(f"Amount of categories in the training set: {amount_categories_train}")
print("Top 20 categories in training dataset:")
for idx, (cat_token, amount) in enumerate(cats_sorted_train[:20], start=1):
    print(f"{idx}. {cat_token}: {amount}")

# Count how often each category occurs in the validation dataset.
cat_mappings = coco_images_val.cat_mappings
coco = coco_images_val.coco
category_counts_val = {}
for img_id in coco_images_val.img_ids:
    anns = coco.loadAnns(coco.getAnnIds(img_id))
    for ann in anns:
        cat = coco.loadCats(ann["category_id"])[0]
        for cat_token in cat["name"].split():
            if cat_token in cat_mappings["stoi"]:
                if cat_token not in category_counts_val:
                    category_counts_val[cat_token] = 0
                category_counts_val[cat_token] += 1
amount_categories_val = len(category_counts_val)
cats_sorted_val = sorted(category_counts_val.items(), key=lambda x: x[1],
                         reverse=True)
print(f"Amount of categories in the validation set: {amount_categories_val}")
print("Top 20 categories in validation dataset:")
for idx, (cat_token, amount) in enumerate(cats_sorted_val[:20], start=1):
    print(f"{idx}. {cat_token}: {amount}")

# Print totals.
amount_images = amount_images_train + amount_images_val
print(f"Total amount of images: {amount_images}")
amount_instances = amount_instances_train + amount_instances_val
print(f"Total amount of instances: {amount_instances}")
amount_categories = len(set(category_counts_train.keys()) |
                        set(category_counts_val.keys()))
print(f"Total amount of categories: {amount_categories}")
