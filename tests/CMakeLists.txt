enable_testing()
find_package(GTest)
if(NOT GTest_FOUND)
    message(WARNING "GTest not found; unit tests will be skipped")
    return()
endif()

add_executable(tests
    test_forward.cpp
    test_plasticity.cpp
    test_reward.cpp
    test_structural.cpp
)

target_link_libraries(tests PRIVATE GTest::gtest_main bb_core)
if(BB_WITH_CUDA)
    target_link_libraries(tests PRIVATE bb_cuda)
endif()
add_test(NAME all_tests COMMAND tests)
