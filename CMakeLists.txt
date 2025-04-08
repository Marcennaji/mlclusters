cmake_minimum_required(VERSION 3.16.3)

project(mlclusters CXX)

# Explicitly set MPI paths
set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
set(MPIEXEC "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")

# Ensure CMake finds MPI
find_package(MPI REQUIRED)

# Debug output to verify MPI detection
if(MPI_FOUND)
    message(STATUS "MPI found:")
    message(STATUS "  MPI CXX Include Path: ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "  MPI CXX Libraries: ${MPI_CXX_LIBRARIES}")
    message(STATUS "  MPI Executable: ${MPIEXEC}")
else()
    message(FATAL_ERROR "MPI not found!")
endif()




# Use Khiops 
include(FetchContent)
FetchContent_Declare(
    khiops
    GIT_REPOSITORY "https://github.com/KhiopsML/khiops.git"
    GIT_TAG "10.7.0-b.0")

set(BUILD_JARS ON CACHE BOOL "Enable building JAR files" FORCE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
set(SET_MPI "SET_MPI_SYSTEM_WIDE")

FetchContent_MakeAvailable(khiops)


file(GLOB files  ${PROJECT_SOURCE_DIR}/src/*cpp  ${PROJECT_SOURCE_DIR}/src/*cc ${PROJECT_SOURCE_DIR}/src/*h )
add_executable(mlclusters  ${files} )
set_khiops_options(mlclusters)

# Link with Khiops libraries
target_link_libraries(mlclusters KMDRRuleLibrary KWLearningProblem)


configure_file(${PROJECT_SOURCE_DIR}/packaging/windows/mlclusters_env.cmd.in ${PROJECT_SOURCE_DIR}/packaging/windows/mlclusters_env.cmd @ONLY
               NEWLINE_STYLE CRLF)