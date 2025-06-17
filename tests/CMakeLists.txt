enable_testing()
find_package(GTest REQUIRED)

add_executable(tests
    test_forward.cpp
    test_plasticity.cpp
    test_reward.cpp
    test_structural.cpp
)

target_link_libraries(tests PRIVATE GTest::gtest_main bb_core)
add_test(NAME all_tests COMMAND tests)
