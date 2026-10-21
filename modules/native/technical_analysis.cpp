#include <pybind11/pybind11.h>
#include <cmath>

int add(int i, int j) {
    return i + j;
}






PYBIND11_MODULE(_technical_analysis_cpp, m) {
    m.doc() = "Native technical analysis helpers for TradeSmart Analytics";
    m.def("add", &add, "A function which adds two numbers");
}
