readonly train_data_link="https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip"
readonly style_data_link="https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Shinkai.tar.gz"

echo $(date)
wget -q "${train_data_link}"
echo "Downloaded Training Data"
unzip -qq dataset.zip
rm -rf Hayao Paprika SummerWar Shinkai
wget -q "${style_data_link}"
echo "Downloaded Training Data"
tar -xzf Shinkai.tar.gz
mkdir dataset
mv train_photo Shinkai val test dataset
rm Shinkai.tar.gz dataset.zip
python edge_smooth.py

echo $(date) "Done"
