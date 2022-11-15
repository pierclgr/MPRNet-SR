# install repo packages
echo "Installing packages for colab..."
cd /content/MPRNet-SR
pip install -e .
pip install -r requirements.txt
echo "Done!"

# copy datasets zip files
echo "Copying the datasets .zip files from Google Drive (may take a while)..."
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/div2k.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/Set5.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/Set14.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/BSD100.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/Urban100.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/Manga109.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/SunHays80.zip /content/MPRNet-SR/data/
mkdir -p /content/MPRNet-SR/data/ && cp /content/drive/MyDrive/Colab\ Notebooks/ML4CV/historical.zip /content/MPRNet-SR/data/
echo "Done!"

# unzip dataset zip files
echo "Unzipping the .zip files (may take a while)..."
unzip -qq /content/MPRNet-SR/data/div2k.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/Set5.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/Set14.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/BSD100.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/Urban100.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/Manga109.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/SunHays80.zip -d /content/MPRNet-SR/data/
unzip -qq /content/MPRNet-SR/data/historical.zip -d /content/MPRNet-SR/data/
echo "Done!"

# deleting the copied zip files to free space
echo "Deleting .zip files to free space..."
rm /content/MPRNet-SR/data/div2k.zip
rm /content/MPRNet-SR/data/Set5.zip
rm /content/MPRNet-SR/data/Set14.zip
rm /content/MPRNet-SR/data/BSD100.zip
rm /content/MPRNet-SR/data/Urban100.zip
rm /content/MPRNet-SR/data/Manga109.zip
rm /content/MPRNet-SR/data/SunHays80.zip
rm /content/MPRNet-SR/data/historical.zip
echo "Done!"
