.. _install:

************
Installation
************

Viziphant requires Python 3.6, 3.7, or 3.8.

.. tabs::

    .. tab:: Stable release version

        The easiest way to install Viziphant is via `pip
        <http://pypi.python.org/pypi/pip>`_:

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

              git clone git://github.com/INM-6/viziphant.git
              cd viziphant
              pip install -e .



Dependencies
------------

Viziphant relies on the following packages (automatically installed when you
run ``pip install viziphant``):

    * `elephant <https://pypi.org/project/elephant/>`_ - Electrophysiology Analysis Toolkit
    * `quantities <http://pypi.python.org/pypi/quantities>`_ - support for physical quantities with units (mV, ms, etc.)
    * `neo <http://pypi.python.org/pypi/neo>`_ - electrophysiology data manipulations
