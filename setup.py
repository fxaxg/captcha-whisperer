"""Build hooks that keep the retraining weights out of the wheel.

The ``captcha_crnn_weights.npz`` file is only needed to retrain and re-export the
model. Everyday inference uses the ONNX graph alone, so shipping the 19 MB weight
file in the wheel would double the download for no benefit.

setuptools cannot express "sdist yes, wheel no" through ``package_data`` and
``exclude_package_data``: whatever the exclusion says applies to wheels built
from a source tree and from an sdist alike. Dropping the file from the wheel
here, while leaving the sdist and MANIFEST.in untouched, is the only reliable
way to get both behaviours.
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

EXCLUDED_SUFFIXES = (".npz",)


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [m for m in modules if not m[1].endswith(EXCLUDED_SUFFIXES)]

    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        return [f for f in files if not f.endswith(EXCLUDED_SUFFIXES)]


setup(cmdclass={"build_py": build_py})
