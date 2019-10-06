# -*- coding: utf-8 -*-

from ast import parse
from os import path
from platform import python_version_tuple

from setuptools import setup, find_packages

if python_version_tuple()[0] == '3':
    imap = map
    ifilter = filter
else:
    from itertools import imap, ifilter

if __name__ == '__main__':
    package_name = 'keras_squeeze_excite_network'

    with open(path.join(package_name, '__init__.py')) as f:
        __author__, __version__ = imap(
            lambda buf: next(imap(lambda e: e.value.s, parse(buf).body)),
            ifilter(lambda line: line.startswith('__version__') or line.startswith('__author__'), f)
        )

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        install_requires=['pyyaml'],
        test_suite='{package_name}.tests'.format(package_name=package_name),
        packages=find_packages(),
        package_dir={package_name: package_name}
    )
