import setuptools

setuptools.setup(
  name='tf-retinanet',
  version='0.2.0',
  description='TensorFlow implementation of RetinaNet object detection',
  url='https://github.com/open-v/tf-retinanet',
  download_url='https://github.com/open-v/tf-retinanet/tags',
  author='Firdavs Beknazarov',
  author_email='opendeeple@gmail.com',
  packages=setuptools.find_packages(),
  install_requires=['tensorflow', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python'],
  setup_requires=['tensorflow>=2.0.0', 'cython>=0.28', 'numpy>=1.14.0'],
  entry_points={
    'console_scripts': [
      'tf-retinanet=tf_retinanet.bin:main'
    ]
  },
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ],
  license='MIT',
  keywords=['TensorFlow RetinaNet Object detection'],
)
