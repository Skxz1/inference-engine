add_executable(run_tests test_sanity.cpp)
target_include_directories(run_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
