cmake_minimum_required(VERSION 3.8)
project(CPPExampleBot LANGUAGES CXX)

add_subdirectory(RLBotCPP)

# ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

add_executable(CPPExampleBot main.cc examplebot.cc examplebot.h)

target_include_directories(CPPExampleBot PRIVATE
    RLBotCPP/inc
    RLBotCPP/lib/inc
    ${ONNXRUNTIME_DIR}/include
)

if (UNIX)
    target_link_libraries(CPPExampleBot RLBotCPP pthread
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    )
else (UNIX)
    target_link_libraries(CPPExampleBot RLBotCPP
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
    )
endif (UNIX)

target_compile_features(CPPExampleBot PRIVATE cxx_std_17)

# Copy port config
add_custom_command(TARGET CPPExampleBot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/port.cfg"
        $<TARGET_FILE_DIR:CPPExampleBot>)

# Copy ONNX model
add_custom_command(TARGET CPPExampleBot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/model.onnx"
        $<TARGET_FILE_DIR:CPPExampleBot>)

# Copy ONNX Runtime DLL/SO
if (UNIX)
    add_custom_command(TARGET CPPExampleBot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
            $<TARGET_FILE_DIR:CPPExampleBot>)
else()
    add_custom_command(TARGET CPPExampleBot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:CPPExampleBot>)
endif()