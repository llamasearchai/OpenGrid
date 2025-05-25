cmake_minimum_required(VERSION 3.16)
project(OpenGrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6 for desktop application
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Qml Quick)

# Find GoogleTest for C++ tests
find_package(GTest REQUIRED)

# Find spdlog for logging
find_package(spdlog REQUIRED)

enable_testing()

# --- C++ Application (Qt/QML based) ---
# The main application source is expected in src/ or a dedicated app folder like visualization/cpp
# For now, assuming main.cpp for the GUI is in OpenGrid/src/
add_executable(opengrid_app src/main.cpp)

target_link_libraries(opengrid_app 
    PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Qml
    Qt6::Quick
    spdlog::spdlog
)
# If QML files are in a subdirectory, e.g., visualization/qml
# target_sources(opengrid_app PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}/visualization/qml/main.qml
# )
# You might also need to set up Qt resource files (.qrc)

# --- C++ Tests ---
add_executable(opengrid_tests tests/main_test.cpp)
# Add other test source files if any: tests/another_test.cpp

target_link_libraries(opengrid_tests 
    PRIVATE
    GTest::GTest
    GTest::Main
    spdlog::spdlog
    # Link against any C++ libraries from src/ that tests need
    # open_grid_core_lib # Example if you create a library from other C++ sources
)

include(GoogleTest)
gtest_discover_tests(opengrid_tests)

# Placeholder for installing the application
# install(TARGETS opengrid_app DESTINATION bin)

# If you have C++ libraries in src/ that are not part of the main app directly but used by tests or Python,
# define them here. E.g.:
# add_library(open_grid_core_lib src/core_functions.cpp src/another_core.cpp)
# target_include_directories(open_grid_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src) 