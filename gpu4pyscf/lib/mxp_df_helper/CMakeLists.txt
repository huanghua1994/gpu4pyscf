set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

set(mxp_df_helper_src
  mxp_df_helper.cu
)

add_library(mxp_df_helper SHARED ${mxp_df_helper_src})

set_target_properties(mxp_df_helper PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set_target_properties(mxp_df_helper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)