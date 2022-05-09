# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class PyViziphant(PythonPackage):
    """Viziphant is a package for the visualization of the analysis results of electrophysiology data in Python"""

    homepage = "https://viziphant.readthedocs.io/en/latest/"
    pypi     = "viziphant/viziphant-0.1.0.tar.gz"

    # notify when the package is updated.
    maintainers = ['Moritz-Alexander-Kern']

    version('0.2.0', sha256='044b5c92de169dfafd9665efe2c310e917d2c21980bcc9f560d5c727161f9bd8')
    version('0.1.0', sha256='8fd56ec8633f799396dc33fbace95d2553bedb17f680a8c0e97f43b3a629bf6c')

    depends_on('py-setuptools',         type='build')
    depends_on('python@3.7:3.10',       type=('build', 'run'))
    depends_on('py-neo@0.9.0:',         type=('build', 'run'))
    depends_on('py-elephant@0.9.0:',    type=('build', 'run'))
    depends_on('py-numpy@1.18.1:',      type=('build', 'run'))
    depends_on('py-quantities@0.12.1:', type=('build', 'run'))
    depends_on('py-six@1.10.0:',        type=('build', 'run'))
    depends_on('py-matplotlib@3.3.2:',  type=('build', 'run'))
    depends_on('py-seaborn@0.9.0:',     type=('build', 'run'))
