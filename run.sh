rm -rf checkpoints/
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
  --subset-size 20000 \
  --learning-rate 2e-4 \
  --style-weight 1e5 \
  --content-weight 1.0 \
  --tv-weight 1e-6 \
  --mixed-precision
