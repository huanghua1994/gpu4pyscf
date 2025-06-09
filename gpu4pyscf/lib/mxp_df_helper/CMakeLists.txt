project(mxp_df_helper LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CUBLAS_ROOT "/home/huah/scratch.huah_gpu/cuda/cublas-12.9.0.58-fp64emu")
set(CUBLAS_INCLUDE_DIR "${CUBLAS_ROOT}/include")
set(CUBLAS_LIBRARY_DIR "${CUBLAS_ROOT}/lib64")

find_library(CUBLAS_LIB cublas HINTS ${CUBLAS_LIBRARY_DIR} REQUIRED)
find_library(CUBLASLT_LIB cublasLt HINTS ${CUBLAS_LIBRARY_DIR} REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcublas -lcublasLt")

set(mxp_df_helper_src
  mxp_df_helper.cu
)

add_library(mxp_df_helper SHARED ${mxp_df_helper_src})
set_target_properties(mxp_df_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/..
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(mxp_df_helper PRIVATE ${CUBLAS_INCLUDE_DIR})
target_compile_options(mxp_df_helper PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-I${CUBLAS_INCLUDE_DIR}>
)

target_link_options(mxp_df_helper PRIVATE "-Wl,--no-as-needed")
target_link_libraries(mxp_df_helper PRIVATE ${CUBLAS_LIB} ${CUBLASLT_LIB})