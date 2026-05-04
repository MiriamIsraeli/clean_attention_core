cmake_minimum_required(VERSION 3.14)
project(CleanAttentionCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(clean_attention_demo
    main.cpp
    lms_filter.cpp
    fxlms_filter.cpp
    noise_detector.cpp
)
