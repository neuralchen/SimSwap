#!/bin/sh
wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
unzip ./checkpoints.zip  -d ./checkpoints
rm checkpoints.zip
wget --no-check-certificate "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" -O antelope.zip
mkdir -p insightface_func/models
unzip ./antelope.zip -d ./insightface_func/models/
rm antelope.zip
