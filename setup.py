# -*- coding: utf-8 -*-

import os

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__),
                       "viziphant", "VERSION")) as version_file:
    version = version_file.read().strip()

with open("README.md") as f:
    long_description = f.read()
with open('requirements/requirements.txt') as fp:
    install_requires = fp.read()


setup(
    name="viziphant",
    version=version,
    packages=['viziphant'],
    include_package_data=True,
    install_requires=install_requires,
    author="Viziphant authors and contributors",
    author_email="contact@python-elephant.org",
    description="Viziphant is a package for the visualization of the analysis"
                "results of electrophysiology data in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url='https://github.com/INM-6/viziphant',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)