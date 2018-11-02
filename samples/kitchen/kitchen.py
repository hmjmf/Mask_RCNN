import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CLASS_NAME = ["BG","plate","cauliflower"]

class KitchenConfig(Config):
  """Configuration for training on the toy  dataset.
  Derives from the base Config class and overrides some values.
  """
  # Give the configuration a recognizable name
  NAME = "kitchen"

  # We use a GPU with 12GB memory, which can fit two images.
  # Adjust down if you use a smaller GPU.
  IMAGES_PER_GPU = 2

  # Number of classes (including background)
  NUM_CLASSES = 1 + len(CLASS_NAME[1:])  # Background + balloon

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # Skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################


class KitchenDataset(utils.Dataset):
  def load_kitchen(self, dataset_dir, subset):
    """Load a subset of the Balloon dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # Add classes.
    for i in range(1,len(CLASS_NAME)):
      print CLASS_NAME[i]
      self.add_class("kitchen", i, CLASS_NAME[i])


    # Train or validation dataset?

    dataset_dir = os.path.join(dataset_dir, subset)

    # Load annotations
    # VGG Image Annotator (up to version 1.6) saves each image in the form:
    # {"filename":"s11-d01-cam-002.avi_2345.jpg","size":22154,
    #  "regions":[
    #    {"shape_attributes":
    #       {"name":"polygon",
    #        "all_points_x":[40,62,71,65,52,34,35],
    #        "all_points_y":[163,159,169,180,183,175,169]},
    #     "region_attributes":{"name":"plate"}},
    #    {"shape_attributes":
    #       {"name":"polygon",
    #        "all_points_x":[2,21,31,20,3,2],
    #        "all_points_y":[171,170,182,191,196,192]},
    #     "region_attributes":{"name":"plate"}}],
    #  "file_attributes":{}}
    # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #   'regions': {
    #       '0': {
    #           'region_attributes': {},
    #           'shape_attributes': {
    #               'all_points_x': [...],
    #               'all_points_y': [...],
    #               'name': 'polygon'}},
    #       ... more regions ...
    #   },
    #   'size': 100202
    # }
    # We mostly care about the x and y coordinates of each region
    # Note: In VIA 2.0, regions was changed from a dict to a list.
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    for a in annotations:
      # Get the x, y coordinaets of points of the polygons that make up
      # the outline of each object instance. These are stores in the
      # shape_attributes (see json format above)
      # The if condition is needed to support VIA versions 1.x and 2.x.
      if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
        labels = [r['region_attributes']["name"] for r in a['regions'].values()]
      else:
        polygons = [r['shape_attributes'] for r in a['regions']]
        labels = [r['region_attributes']["name"] for r in a['regions']]

        # load_mask() needs the image size to convert polygons to masks.
      # Unfortunately, VIA doesn't include it in JSON, so we must read
      # the image. This is only managable since the dataset is tiny.
      image_path = os.path.join(dataset_dir, a['filename'])
      image = skimage.io.imread(image_path)
      height, width = image.shape[:2]
      self.add_image(
        "kitchen",
        image_id=a['filename'],  # use file name as a unique image id
        path=image_path,
        width=width, height=height,
        polygons=polygons,labels = labels)

  def load_mask(self, image_id):
    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "kitchen":
      return super(self.__class__, self).load_mask(image_id)

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)
    for i, p in enumerate(info["polygons"]):
      # Get indexes of pixels inside the polygon and set them to 1
      rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
      mask[rr, cc, i] = 1

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    labels = [CLASS_NAME.index(i) for i in image_info["labels"]]
    return mask.astype(np.bool), np.array(labels,dtype=np.int64)

  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "kitchen":
      return info["labels"]
    else:
      super(self.__class__, self).image_reference(image_id)

def train(model):
  """Train the model."""
  # Training dataset.
  dataset_train = KitchenDataset()
  dataset_train.load_kitchen(args.dataset, ".")
  dataset_train.prepare()

  # Validation dataset
  dataset_val = KitchenDataset()
  dataset_val.load_kitchen(args.dataset, ".")
  dataset_val.prepare()

  # *** This training schedule is an example. Update to your needs ***
  # Since we're using a very small dataset, and starting from
  # COCO trained weights, we don't need to train too long. Also,
  # no need to train all layers, just the heads should do it.
  # no need to train all layers, just the heads should do it.
  print("Training network heads")
  model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE,
              epochs=10,
              layers='heads')
def splash(model):
  import random
  while 1:
    filename = "images/" + random.choice(os.listdir("images"))


    # Run model detection and generate the color splash effect
    print("Running on {}".format(filename))
    # Read image
    image = skimage.io.imread(filename)
    # Detect objects
    r = model.detect([image], verbose=0)[0]
    # Color splash
    print r['class_ids']

    if 2 in r['class_ids']:
      visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                              CLASS_NAME, r['scores'])





if __name__ == '__main__':
  import argparse

  # Parse command line arguments
  parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect balloons.')
  parser.add_argument("command",
                      metavar="<command>",
                      help="'train' or 'splash'")
  parser.add_argument('--dataset', required=False,
                      metavar="/path/to/balloon/dataset/",
                      help='Directory of the Balloon dataset')
  parser.add_argument('--weights', required=True,
                      metavar="/path/to/weights.h5",
                      help="Path to weights .h5 file or 'coco'")
  parser.add_argument('--logs', required=False,
                      default=DEFAULT_LOGS_DIR,
                      metavar="/path/to/logs/",
                      help='Logs and checkpoints directory (default=logs/)')
  args = parser.parse_args()

  # Validate arguments
  if args.command == "train":
    assert args.dataset, "Argument --dataset is required for training"


  print("Weights: ", args.weights)
  print("Dataset: ", args.dataset)
  print("Logs: ", args.logs)

  # Configurations
  if args.command == "train":
    config = KitchenConfig()
  else:
    class InferenceConfig(KitchenConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1
    config = InferenceConfig()
  config.display()

  # Create model
  if args.command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)
  else:
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)

  # Select weights file to load
  if args.weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
      utils.download_trained_weights(weights_path)
  elif args.weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
  elif args.weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
  else:
    weights_path = args.weights

  # Load weights
  print("Loading weights ", weights_path)
  if args.weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
      "mrcnn_class_logits", "mrcnn_bbox_fc",
      "mrcnn_bbox", "mrcnn_mask"])
  else:
    model.load_weights(weights_path, by_name=True)

  # Train or evaluate
  if args.command == "train":
    train(model)
  elif args.command == "splash":
    splash(model)

  else:
    print("'{}' is not recognized. "
          "Use 'train' or 'splash'".format(args.command))