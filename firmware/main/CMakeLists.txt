idf_component_register(SRCS "app_main.cpp" "float_ref.c" INCLUDE_DIRS "."
                       EMBED_FILES "tinycnn_int8.espdl" "float_weights.bin" "testset.bin")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
set_source_files_properties(float_ref.c PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math")
