# OpenGrid Visualization (Qt/C++ Desktop Application)

This directory is intended for components related to the OpenGrid Qt/C++ desktop application, primarily QML files and C++ classes that directly support the UI logic if separated from the core `src`.

## Overview
- The main C++ application entry point is in `OpenGrid/src/main.cpp`.
- The UI is built using Qt 6.5.2 and QML.
- Interactive grid design, simulation visualization, and results display are handled by this desktop application.

## QML Files
- QML files defining the UI structure and behavior will reside here or in a subdirectory (e.g., `qml/`).
- Example: `main.qml` (referenced by `src/main.cpp`).

## C++ UI Logic
- C++ classes that expose data models to QML, handle user interactions, or manage UI state might be placed here or within `src/` if tightly coupled with core C++ logic.

## Building and Running
- The desktop application is built using CMake as defined in `OpenGrid/CMakeLists.txt`.
- After building (e.g., in a `build/` directory), the application executable (e.g., `opengrid_app`) can be run. 