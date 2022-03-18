"""Setup script for this repository."""
from os.path import dirname
from os.path import realpath
from setuptools import setup

from ddpgfd.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='DDPGfD',
    version=__version__,
    install_requires=_read_requirements_file(),
)
