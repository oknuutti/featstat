__author__ = "Olli Knuuttila"
__date__ = "$Feb 2, 2023 1:27:11 PM$"

from setuptools import setup, find_packages

setup(
    name='featstat',
    version='1.0',
    packages=find_packages(include=['featstat*']),
    include_package_data=True,

    # Declare your packages' dependencies here, for eg:
    install_requires=['numpy', 'scipy', 'numba', 'matplotlib', 'python-dateutil', 'tqdm',
                      'opencv-python', 'opencv-contrib-python',      # conda: opencv, pip: opencv-python
                      'numpy-quaternion'],                           # conda: quaternion, pip: numpy-quaternion

    author=__author__,
    author_email='olli.knuuttila@gmail.com',

    summary='Feature tracking statistics',
    url='https://github.com/oknuutti/featstat',
    license='MIT',
)