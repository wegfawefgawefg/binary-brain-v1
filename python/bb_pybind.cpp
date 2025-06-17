#include <pybind11/pybind11.h>
#include "bb_core/network.hpp"

PYBIND11_MODULE(_binarybrain, m) {
    namespace py = pybind11;
    py::class_<bb::Network>(m, "Network")
        .def(py::init<unsigned>())
        .def("step", &bb::Network::step)
        .def_property_readonly("reward", &bb::Network::reward);
}
