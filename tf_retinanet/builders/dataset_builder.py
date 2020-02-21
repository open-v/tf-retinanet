from ..datasets import CSVDatasetBuilder

builders = {
  'csv': CSVDatasetBuilder
}
def build(name, *args, **kwargs):
  if not name in builders:
    raise ValueError('unsupported dataset builder')
  builder = builders[name]
  with builder(*args, **kwargs) as dataset:
    dataset.convert()
