project(mxp_df_helper LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcublas -lcublasLt")

set(mxp_df_helper_src
  mxp_df_helper.cu
)

add_library(mxp_df_helper SHARED ${mxp_df_helper_src})
set_target_properties(mxp_df_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)

find_library(CUBLASLT_LIB cublasLt HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if (NOT CUBLASLT_LIB)
  message(FATAL_ERROR "Could not find libcublasLt.so")
endif()
target_link_options(mxp_df_helper PRIVATE "-Wl,--no-as-needed")
target_link_libraries(mxp_df_helper PRIVATE cublas)
target_link_libraries(mxp_df_helper PRIVATE cublasLt)