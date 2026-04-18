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
  --batch-size 8 \
  --epochs 2 \
  --subset-size 10000 \
  --learning-rate 2e-4 \
  --style-weight 1e5 \
  --content-weight 1.0 \
  --tv-weight 1e-6 \
#  --mixed-precision

# After training, stylize the images in the content_examples directory and save the results to outputs/mosaic_custom
python stylize.py \
  --model checkpoints/mosaic_custom_final.pth \
  --input data/content_examples \
  --output-dir outputs/mosaic_custom