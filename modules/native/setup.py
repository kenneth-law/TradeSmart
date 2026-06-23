from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "_technical_analysis_cpp",
        ["technical_analysis.cpp"],
        cxx_std=17,
    ),
    Pybind11Extension(
        "_backtesting_cpp",
        ["backtesting.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="tradesmart-native",
    version="0.1.0",
    description="Native C++ extensions for TradeSmart Analytics",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
