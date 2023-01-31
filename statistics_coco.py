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

# Print statistics about images and instances.
print(f"Amount of images in the training set: {amount_images_train}")
print(f"Amount of images in the validation set: {amount_images_val}")
print(f"Amount of instances in the training set: {amount_instances_train}")
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

# Aggregate the category counts.
category_counts = {}
for cat_token, count in category_counts_train.items():
    if cat_token not in category_counts:
        category_counts[cat_token] = 0
    category_counts[cat_token] = count
for cat_token, count in category_counts_val.items():
    if cat_token not in category_counts:
        category_counts[cat_token] = 0
    category_counts[cat_token] += count
cats_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
print("Top 20 categories in the dataset:")
for idx, (cat_token, amount) in enumerate(cats_sorted[:20], start=1):
    print(f"{idx}. {cat_token}: {amount}")

# Print amount of instances in top x categories.
instances_top5_cats = sum([count for _, count in cats_sorted[:5]])
instances_top10_cats = sum([count for _, count in cats_sorted[:10]])
instances_top20_cats = sum([count for _, count in cats_sorted[:20]])
instances_all_cats = sum(category_counts.values())
r5 = instances_top5_cats / instances_all_cats
r10 = instances_top10_cats / instances_all_cats
r20 = instances_top20_cats / instances_all_cats
print(f"Instances in the top 5 cats: {r5 * 100:.1f}% (recall {r5})")
print(f"Instances in the top 10 cats: {r10 * 100:.1f}% (recall {r10})")
print(f"Instances in the top 20 cats: {r20 * 100:.1f}% (recall {r20})")

# Print totals.
amount_images = amount_images_train + amount_images_val
print(f"Total amount of images: {amount_images}")
amount_instances = amount_instances_train + amount_instances_val
print(f"Total amount of instances: {amount_instances}")
amount_categories = len(set(category_counts_train.keys()) |
                        set(category_counts_val.keys()))
print(f"Total amount of categories: {amount_categories}")
