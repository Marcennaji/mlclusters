cmake_minimum_required(VERSION 3.16.3)

project(mlclusters CXX)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)

set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
set(MPIEXEC "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")

# Use Khiops 
include(FetchContent)
FetchContent_Declare(
    khiops
    GIT_REPOSITORY "https://github.com/KhiopsML/khiops.git"
    GIT_TAG "10.7.0-b.0")

set(BUILD_JARS ON CACHE BOOL "Enable building JAR files" FORCE)

FetchContent_MakeAvailable(khiops)


file(GLOB files  ${PROJECT_SOURCE_DIR}/src/*cpp  ${PROJECT_SOURCE_DIR}/src/*cc ${PROJECT_SOURCE_DIR}/src/*h )
add_executable(mlclusters  ${files} )
set_khiops_options(mlclusters)

# Link with Khiops libraries
target_link_libraries(mlclusters KMDRRuleLibrary KWLearningProblem)
