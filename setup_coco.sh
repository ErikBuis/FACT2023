# This script is used to download the COCO dataset and extract the
# images and annotations.

coco_dir="./data/coco"

# Download the COCO dataset.
echo "Downloading the COCO dataset..."
wget http://images.cocodataset.org/zips/train2017.zip -P "$coco_dir"
wget http://images.cocodataset.org/zips/val2017.zip -P "$coco_dir"

# Extract the images.
echo "Extracting the images..."
unzip "$coco_dir"/train2017.zip -d "$coco_dir"
unzip "$coco_dir"/val2017.zip -d "$coco_dir"

# Remove the zip files.
rm "$coco_dir"/train2017.zip
rm "$coco_dir"/val2017.zip

# Download the annotations.
echo "Downloading the annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P "$coco_dir"

# Extract the annotations.
echo "Extracting the annotations..."
unzip "$coco_dir"/annotations_trainval2017.zip -d "$coco_dir"

# Remove the zip file.
rm "$coco_dir"/annotations_trainval2017.zip

# Run the script to pre-process the dataset.
echo "Preprocessing the dataset..."
python3 preprocess_coco.py

# Print confirmation message.
echo -e "\n>>> COCO setup script finished <<<"
