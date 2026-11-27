cmake_minimum_required(VERSION 3.16)
project(astro_stacker VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASTRO_NATIVE_ARCH "Optimize for the host CPU (-march=native); disable for portable/distro packages" ON)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBRAW REQUIRED IMPORTED_TARGET libraw_r)

add_executable(astro_stacker
    cpp_stacker/src/raw_loader.cpp
    cpp_stacker/src/calibration.cpp
    cpp_stacker/src/demosaic.cpp
    cpp_stacker/src/star_detector.cpp
    cpp_stacker/src/registration.cpp
    cpp_stacker/src/flat_field.cpp
    cpp_stacker/src/integrator.cpp
    cpp_stacker/src/io_export.cpp
    cpp_stacker/src/main.cpp
)

target_include_directories(astro_stacker PRIVATE cpp_stacker/include)

target_compile_options(astro_stacker PRIVATE
    -Wall -Wextra -pedantic
    $<$<AND:$<BOOL:${ASTRO_NATIVE_ARCH}>,$<CXX_COMPILER_ID:GNU,Clang>>:-march=native>
)

target_link_libraries(astro_stacker PRIVATE
    OpenMP::OpenMP_CXX
    PkgConfig::LIBRAW
)

include(GNUInstallDirs)
install(TARGETS astro_stacker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
