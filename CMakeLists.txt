cmake_minimum_required(VERSION 3.16.3)

project(mlclusters CXX)

# Use Khiops 
include(FetchContent)
FetchContent_Declare(
    khiops
    GIT_REPOSITORY "https://github.com/KhiopsML/khiops.git"
    GIT_TAG "10.7.0-b.0")

FetchContent_MakeAvailable(khiops)


file(GLOB files  ${PROJECT_SOURCE_DIR}/src/*cpp  ${PROJECT_SOURCE_DIR}/src/*cc ${PROJECT_SOURCE_DIR}/src/*h )
add_executable(mlclusters  ${files} )
set_khiops_options(mlclusters)

# Link with Khiops libraries
target_link_libraries(mlclusters KMDRRuleLibrary KWLearningProblem)