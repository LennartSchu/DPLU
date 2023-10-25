add_executable(test_eigen eigen/test_eigen.cpp)
target_link_libraries(test_eigen PRIVATE Eigen3::Eigen spdlog::spdlog)

add_executable(test_dplu dplu/test_dplu.cpp)
target_include_directories(test_dplu PRIVATE ${PROJECT_SOURCE_DIR}/dplu/include)
target_link_libraries(test_dplu PRIVATE Eigen3::Eigen spdlog::spdlog)

enable_testing()

add_test(NAME test_eigen COMMAND $<TARGET_FILE:test_eigen>)
add_test(NAME test_dplu COMMAND $<TARGET_FILE:test_dplu>)