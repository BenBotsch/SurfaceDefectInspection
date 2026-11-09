cmake_minimum_required(VERSION 3.16)
project(sdi_classify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_library(sdi_classifier src/classifier.cpp)
target_include_directories(sdi_classifier PUBLIC src)
target_link_libraries(sdi_classifier PUBLIC ${OpenCV_LIBS})
target_compile_options(sdi_classifier PRIVATE -Wall -Wextra -Werror)

add_executable(sdi-classify src/main.cpp)
target_link_libraries(sdi-classify PRIVATE sdi_classifier)

enable_testing()
add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity PRIVATE sdi_classifier)
# The fixture comes from `sdi parity-fixture <dir>`; pass its directory with -DSDI_FIXTURE=<dir>.
set(SDI_FIXTURE "" CACHE PATH "directory written by `sdi parity-fixture`")
if(SDI_FIXTURE)
  add_test(NAME parity COMMAND test_parity ${SDI_FIXTURE})
  add_test(NAME bench COMMAND sdi-classify --bench 3 ${SDI_FIXTURE}/model.onnx
                                ${SDI_FIXTURE}/image.png)
  set_tests_properties(bench PROPERTIES PASS_REGULAR_EXPRESSION "\"median_ms\": [0-9]")
endif()
