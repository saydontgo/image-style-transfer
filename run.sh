# Remove previous checkpoints
rm -rf checkpoints/

# Train the model with the specified parameters
python train.py \
  --dataset data/coco_train2014 \
  --style-image data/style_images/mosaic.jpg \
  --preview-dir data/content_examples \
  --output-dir checkpoints \
  --run-name mosaic_custom \
  --image-size 256 \
  --style-size 512 \
  --batch-size 32 \
  --epochs 4 \
  --subset-size 0 \
  --learning-rate 2e-4 \
  --style-weight 1e6 \
  --content-weight 1.0 \
  --tv-weight 1e-6 \
#  --mixed-precision

# After training, stylize the images in the content_examples directory and save the results to outputs/mosaic_custom
# python stylize.py \
#   --model checkpoints/mosaic_custom_final.pth \
#   --input data/content_examples \
#   --output-dir outputs/mosaic_custom

# Compare the stylized images from the custom model with those from the pretrained mosaic model and save the results to outputs/compare_mosaic
python compare_models.py \
  --content-dir data/content_examples \
  --baseline-model external_models/mosaic.pth \
  --custom-model checkpoints/mosaic_custom_final.pth \
  --baseline-label pretrained_mosaic \
  --custom-label my_mosaic \
  --output-dir outputs/compare_mosaic