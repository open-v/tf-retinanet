import sys
import argparse
import tf_retinanet

parser = argparse.ArgumentParser(description='RetinaNet object detection')
parser.add_argument('--train', help='Training a model', dest='train', action='store_true')
parser.add_argument('--dataset', help='Path to dataset', type=str)
parser.add_argument('--convert', help='Model convertation', dest='convert', action='store_true')
parser.add_argument('--backbone', help='Backbone model used by RetinaNet', default='mobilenet_v1', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', default=5, type=int)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
parser.add_argument('--config', help='Path to a configuration parameters .ini file')
parser.add_argument('--checkpoints', help='Path to checkpoints')

def main(args=None):
  args = parser.parse_args(sys.argv[1:])
  if args.train:
    model = tf_retinanet.build(args.backbone)
    if not args.dataset:
      args.dataset = input('Input path to dataset: ')
    if not args.checkpoints:
      args.checkpoints = input('Input path to store checkpoints: ')
    dataset = tf_retinanet.data.read(args.dataset)
    tf_retinanet.train(model, dataset, epochs=args.epochs, lr=args.lr, checkpoints=args.checkpoints)
