add_library(bb_core STATIC
    neuron.cpp
    synapse.cpp
    reward_bus.cpp
    sensor_pool.cpp
    actuator_pool.cpp
    network.cpp
    plasticity.cpp
    structural_plasticity.cpp
)

target_include_directories(bb_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/cuda
)

if(BB_WITH_CUDA)
    target_compile_definitions(bb_core PUBLIC BB_WITH_CUDA)
endif()

add_dependencies(bb_core bb_headers)
