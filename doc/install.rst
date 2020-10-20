.. _install:

************
Installation
************

The easiest way to install Viziphant is by creating a conda environment, followed by ``pip install viziphant``.
Below is the explanation of how to proceed with these two steps.


Prerequisites
=============

Viziphant requires Python_ 3.6, 3.7, or 3.8.

.. tabs::


    .. tab:: (recommended) Conda (Linux/MacOS/Windows)

        1. Create your conda environment (e.g., `viziphant_env`):

           .. code-block:: sh

              conda create --name viziphant_env python=3.7 numpy scipy tqdm

        2. Activate your environment:

           .. code-block:: sh

              conda activate viziphant_env


    .. tab:: Debian/Ubuntu

        Open a terminal and run:

        .. code-block:: sh

           sudo apt-get install python-pip python-numpy python-scipy python-pip python-six python-tqdm



Installation
============

.. tabs::


    .. tab:: Stable release version

        The easiest way to install Viziphant is via pip_:

           .. code-block:: sh

              pip install viziphant

        To upgrade to a newer release use the ``--upgrade`` flag:

           .. code-block:: sh

              pip install --upgrade viziphant

        If you do not have permission to install software systemwide, you can
        install into your user directory using the ``--user`` flag:

           .. code-block:: sh

              pip install --user viziphant


    .. tab:: Development version

        If you have `Git <https://git-scm.com/>`_ installed on your system,
        it is also possible to install the development version of Viziphant.

        1. Before installing the development version, you may need to uninstall
           the previously installed version of Viziphant:

           .. code-block:: sh

              pip uninstall viziphant

        2. Clone the repository install the local version:

           .. code-block:: sh

              git clone git://github.com/INM-6/viziphant-new.git
              cd viziphant
              pip install -e .



Dependencies
------------

The following packages are required to use Viziphant (refer to requirements_ for the exact package versions):

    * numpy_ - fast array computations
    * quantities_ - support for physical quantities with units (mV, ms, etc.)
    * matplotlib_ - 2D plotting library
    * seaborn_ - statistical data visualization
    * six_ - Python 2 and 3 compatibility utilities

These packages are automatically installed when you run ``pip install viziphant``.


.. _`Python`: http://python.org/
.. _`numpy`: http://www.numpy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _matplotlib: https://matplotlib.org/
.. _seaborn: https://seaborn.pydata.org/
.. _`neo`: http://pypi.python.org/pypi/neo
.. _`pip`: http://pypi.python.org/pypi/pip
.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _`Conda environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _`tqdm`: https://pypi.org/project/tqdm/
.. _`six`: https://pypi.org/project/six/
.. _requirements: https://github.com/INM-6/viziphant-new/blob/master/requirements/requirements.txt
.. _PyPI: https://pypi.org/
