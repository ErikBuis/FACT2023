# This script is used to download the Visual Genome dataset and extract the
# images and annotations.

# Download the Visual Genome dataset
echo "Downloading the Visual Genome dataset..."
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -p data/vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -p data/vg

# Extract the images
echo "Extracting the images..."
unzip data/vg/images.zip -d data/vg
unzip data/vg/images2.zip -d data/vg

# Combine the images into one folder
mv data/vg/VG_100K_2/* data/vg/VG_100K

# Remove the empty folders and zip files
rm -r data/vg/VG_100K_2
rm data/vg/images.zip
rm data/vg/images2.zip

# Download the annotations
echo "Downloading the annotations..."
wget http://visualgenome.org/static/data/dataset/objects.json.zip -p data/vg

# Extract the annotations
unzip data/vg/objects.json.zip -d data/vg/vg_objects.json

# Remove the zip file
rm data/vg/objects.json.zip

# Run the script to create the dataset
echo "Creating the dataset..."
python3 preprocess_vg.py
