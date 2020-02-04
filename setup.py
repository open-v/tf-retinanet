import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

class BuildExtension(setuptools.Command):
  description = DistUtilsBuildExt.description
  user_options = DistUtilsBuildExt.user_options
  boolean_options = DistUtilsBuildExt.boolean_options
  help_options = DistUtilsBuildExt.help_options

  def __init__(self, *args, **kwargs):
    from setuptools.command.build_ext import build_ext as SetupToolsBuildExt
    self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)
  
  def __getattr__(self, name):
    return getattr(self._command, name)
  
  def __setattr__(self, name, value):
    setattr(self._command, name, value)

  def initialize_options(self, *args, **kwargs):
    return self._command.initialize_options(*args, **kwargs)
  
  def finalize_options(self, *args, **kwargs):
    ret = self._command.finalize_options(*args, **kwargs)
    import numpy
    self.include_dirs.append(numpy.get_include())
    return ret
  
  def run(self, *args, **kwargs):
    return self._command.run(*args, **kwargs)

extensions = [
  Extension(
    'tf_retinanet.builders.compute_overlap',
    ['tf_retinanet/builders/compute_overlap.pyx']
  ),
]

setuptools.setup(
  name = 'tf-retinanet',
  version = '0.1.0',
  description = 'TensorFlow implementation of RetinaNet object detection',
  url = 'https://github.com/open-v/tf-retinanet',
  author = 'Firdavs Beknazarov',
  author_email = 'opendeeple@gmail.com',
  cmdclass = {'build_ext': BuildExtension},
  packages = setuptools.find_packages(),
  install_requires=['tensorflow', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python'],
  # entry_points = {},
  ext_modules = extensions,
  setup_requires = ['tensorflow>=2.0', 'cython>=0.28', 'numpy>=1.14.0']
)


