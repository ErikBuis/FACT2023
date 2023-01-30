# This script is used to download the Visual Genome dataset and extract the
# images and annotations.

vg_dir="./data/vg"

# Download the Visual Genome dataset.
echo "Downloading the Visual Genome dataset..."
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P "$vg_dir"
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P "$vg_dir"

# Extract the images.
echo "Extracting the images..."
unzip "$vg_dir"/images.zip -d "$vg_dir"
unzip "$vg_dir"/images2.zip -d "$vg_dir"

# Combine the images into one folder.
mv "$vg_dir"/VG_100K_2/* "$vg_dir"/VG_100K

# Remove the empty folder and zip files.
rm -r "$vg_dir"/VG_100K_2
rm "$vg_dir"/images.zip
rm "$vg_dir"/images2.zip

# Download the annotations.
echo "Downloading the annotations..."
wget http://visualgenome.org/static/data/dataset/objects.json.zip -P "$vg_dir"

# Extract the annotations.
echo "Extracting the annotations..."
unzip "$vg_dir"/objects.json.zip -d "$vg_dir"/vg_objects.json

# Remove the zip file.
rm "$vg_dir"/objects.json.zip

# Run the script to pre-process the dataset.
echo "Creating the dataset..."
python3 preprocess_vg.py

# Remove the original annotations file.
rm "$vg_dir"/vg_objects.json

# Print confirmation message.
echo -e "\n>>> VG setup script finished <<<"
