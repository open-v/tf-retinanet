from . import backbone_builder
from ..models import RetinaNet

def build(name, num_classes=None, num_values=None, num_anchors=None, **kwargs):
  backbone = backbone_builder.build(name, **kwargs)
  return RetinaNet(backbone, num_classes=num_classes, num_values=num_values, num_anchors=num_anchors)
