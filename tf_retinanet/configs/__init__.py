from .anchors import AnchorsConfig

class Config:
  def __init__(self,
      num_classes=90,
      num_localization_values=4,
      anchors_sizes=None, 
      anchors_strides=None,
      anchors_ratios=None,
      anchors_scales=None
    ):
    self.num_localization_values = num_localization_values
    self.num_classes = num_classes
    self.anchors = AnchorsConfig(
      sizes=anchors_sizes, 
      strides=anchors_strides, 
      ratios=anchors_ratios, 
      scales=anchors_scales
    )

  def get_config(self):
    return {
      'num_classes': self.num_classes,
      'num_localization_values': self.num_localization_values,
      'anchors_sizes': self.anchors.sizes,
      'anchors_strides': self.anchors.strides,
      'anchors_ratios': self.anchors.ratios,
      'anchors_scales': self.anchors.scales
    }

  def __repr__(self):
    return 'Config(num_classes={}, num_localization_values={}, anchors={})'.format(
      self.num_classes,
      self.num_localization_values,
      repr(self.anchors)
    )

config = Config()
def set_config(**kwargs):
  global config
  options = config.get_config()
  options.update(**kwargs)
  config = Config(**options)

