import setuptools

setuptools.setup(
  name='tf-retinanet',
  version='0.2.0',
  description='TensorFlow implementation of RetinaNet object detection',
  url='https://github.com/open-v/tf-retinanet',
  author='Firdavs Beknazarov',
  author_email='opendeeple@gmail.com',
  packages=setuptools.find_packages(),
  install_requires=['tensorflow', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python'],
  setup_requires=['tensorflow>=2.0.0', 'cython>=0.28', 'numpy>=1.14.0'],
  entry_points={
    'console_scripts': [
      'tf-retinanet=tf_retinanet.bin:main'
    ]
  }
)
