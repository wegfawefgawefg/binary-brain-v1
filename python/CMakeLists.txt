find_package(pybind11 REQUIRED)

pybind11_add_module(_binarybrain bb_pybind.cpp)

target_link_libraries(_binarybrain PRIVATE bb_core bb_cuda)
