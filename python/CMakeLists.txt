find_package(pybind11)
if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found; Python bindings will be disabled")
    return()
endif()

pybind11_add_module(_binarybrain bb_pybind.cpp)

target_link_libraries(_binarybrain PRIVATE bb_core)
if(BB_WITH_CUDA)
    target_link_libraries(_binarybrain PRIVATE bb_cuda)
endif()
