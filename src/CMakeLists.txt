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
)

add_dependencies(bb_core bb_headers)
