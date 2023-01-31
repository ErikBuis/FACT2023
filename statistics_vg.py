import os

from image_datasets import VisualGenomeImages, VisualGenomeInstances

root = "./data/vg"
train_frac = 0.9
cat_frac = 0.7

# Count the number of images in the dataset.
vg_images = VisualGenomeImages(
    objs_file=os.path.join(root, "vg_objects_preprocessed.json"),
    root=root,
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=None,
    filter_width=224,
    filter_height=224
)
amount_images = len(vg_images)
p_images_train = train_frac * 100
p_images_val = 100 - train_frac * 100
print(f"Percentage of images in the training set: {p_images_train:.1f}%")
print(f"Percentage of images in the validation set: {p_images_val:.1f}%")
print(f"Total amount of images: {amount_images}")

# Count the number of instances in the dataset.
vg_instances = VisualGenomeInstances(
    objs_file=os.path.join(root, "vg_objects_preprocessed.json"),
    root=root,
    cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
    transform=None,
    filter_width=224,
    filter_height=224
)
amount_instances = len(vg_instances)
p_instances_train = train_frac * 100
p_instances_val = 100 - train_frac * 100
print(f"Percentage of instances in the training set: {p_instances_train:.1f}%")
print(f"Percentage of instances in the validation set: {p_instances_val:.1f}%")
print(f"Total amount of instances: {amount_instances}")

# Count how often each category occurs in the dataset.
cat_mappings = vg_images.cat_mappings
samples = vg_images.samples
category_counts = {}
for sample in samples:
    for cat_token, bboxes in sample["objects"].items():
        if cat_token not in category_counts:
            category_counts[cat_token] = 0
        category_counts[cat_token] += len(bboxes)
amount_cats = len(category_counts)
p_cats_train = cat_frac * 100
p_cats_val = 100 - cat_frac * 100
cats_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

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

# Print statistics about categories.
print(f"Percentage of categories in the training set: {p_cats_train:.1f}%")
print(f"Percentage of categories in the validation set: {p_cats_val:.1f}%")
print(f"Amount of categories in the dataset: {amount_cats}")
print("Top 20 categories in dataset:")
for idx, (cat_token, amount) in enumerate(cats_sorted[:20], start=1):
    print(f"{idx}. {cat_token}: {amount}")
