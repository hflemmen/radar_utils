from setuptools import setup

setup(name='radar_utils',
      version='0.1',
      description='Helper functions to work with radar images.',
      url='https://github.com/hflemmen/radar_utils',
      author='Henrik',
      author_email='henrik.d.flemmen@ntnu.no',
      license='MIT',
      packages=['radar_utils'],
      install_requires=[
            'opencv-python',
            'numpy',
            'polarTransform',
            'matplotlib',
            # 'numba',
      ],
      zip_safe=False)