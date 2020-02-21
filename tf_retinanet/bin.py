import sys
import argparse
import tf_retinanet
from  tf_retinanet import builders

parser = argparse.ArgumentParser(description='RetinaNet object detection')
parser.add_argument('--train', help='Training a model', dest='train', action='store_true')
parser.add_argument('--dataset', help='Path to dataset', type=str)
parser.add_argument('--convert', help='Model convertation', dest='convert', action='store_true')
parser.add_argument('--backbone', help='Backbone model used by RetinaNet', default='mobilenet_v1', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', default=5, type=int)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
parser.add_argument('--config', help='Path to a configuration parameters .ini file')
parser.add_argument('--checkpoints', help='Path to checkpoints')
parser.add_argument('--csv', help='Build dataset from csv', dest='csv', action='store_true')
parser.add_argument('--boxes', help='Path to boxes csv data', type=str)
parser.add_argument('--labels', help='Path to labels csv data', type=str)
parser.add_argument('--coco', help='Path to coco dataset', type=str)
parser.add_argument('--num-shards', help='Number of shards', dest='num_shards', type=int, default=10)
parser.add_argument('--image-min-size', help='Min size of image', dest='image_min_size', type=int, default=200)
parser.add_argument('--image-max-size', help='Max size of image', dest='image_max_size', type=int, default=300)
parser.add_argument('--num-classes', help='Num classes', dest='num_classes', type=int)

def main(args=None):
  args = parser.parse_args(sys.argv[1:])
  # csv dataset building
  if args.csv:
    if not args.boxes:
      args.boxes = input('Input path to boxes csv data: ')
    if not args.labels:
      args.labels = input('Input path to labels csv data: ')
    if not args.dataset:
      args.dataset = input('Input path to dataset: ')
    tf_retinanet.builders.dataset_builder.build(
      'csv', args.dataset, args.boxes, args.labels,
      num_shards=args.num_shards, image_min_size=args.image_min_size, image_max_size=args.image_max_size
    )

  # training
  if args.train:
    model = tf_retinanet.build(args.backbone, num_classes=args.num_classes)
    if not args.dataset:
      args.dataset = input('Input path to dataset: ')
    if not args.checkpoints:
      args.checkpoints = input('Input path to store checkpoints: ')
    if not args.num_classes:
      args.num_classes = int(input('Input num classes: '))
    dataset = tf_retinanet.data.read(args.dataset)
    tf_retinanet.train(model, dataset, num_classes=args.num_classes, epochs=args.epochs,
        lr=args.lr, checkpoints=args.checkpoints)

if __name__ == '__main__':
  main()
