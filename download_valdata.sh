#!/bin/sh
echo 'downloading validation files'
if [ -d "data" ]
then
    echo 'data dir found, continuing'
else
    mkdir "data"
fi

# can't just do recursive wget through the dir, so have to use hardcoded values
wget https://files.warwick.ac.uk/tkillestein/files/gotorb_validation_data/datapack.hkl -P ./data
wget https://files.warwick.ac.uk/tkillestein/files/gotorb_validation_data/gotorb_valmodel_BALDflip_20201030-170220.h5 -P ./data

echo 'complete!'
