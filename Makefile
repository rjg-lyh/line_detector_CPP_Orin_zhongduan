cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
cpp_mks := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:src/%=objs/%)
cu_mks   := $(cu_objs:.cu.o=.cu.mk)

nvcc := /usr/local/cuda/bin/nvcc -ccbin=g++

NO_COLOR=\033[0m
GREEN_COLOR=\033[32;01m
RED_COLOR=\033[31;01m

include_paths := src \
				/usr/local/include/opencv4 \
				/usr/local/cuda/include \
				/usr/local/protobuf-3.11.4-cpp/include \
				/usr/include/aarch64-linux-gnu \
				/usr/local/include             #Eigen、SerialPort

# include_paths := src \
# 				/usr/local/include/opencv4 \
# 				/usr/local/cuda/include \
# 				/usr/local/protobuf-3.11.4-cpp/include \
# 				/usr/local/TensorRT-8.5.3.1/include \
# 				/usr/local/include             #Eigen、SerialPort


library_paths := /usr/local/lib \
				/usr/local/cuda/lib64 \
				/usr/local/protobuf-3.11.4-cpp/lib \
				/usr/lib/aarch64-linux-gnu

# library_paths := /usr/local/lib \
# 				/usr/local/cuda-11.3/lib64 \
# 				/usr/local/protobuf-3.11.4-cpp/lib \
# 				/usr/local/TensorRT-8.5.3.1/lib
				
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

#LIBOPENCV_LIBS := $(shell pkg-config --libs opencv)

run_paths := $(library_paths:%=-Wl,-rpath=%)
include_paths := $(include_paths:%=-I%)
library_paths := $(library_paths:%=-L%)

opencv_ld   := opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_calib3d
cuda_ld     := cudart cudnn
#tensorrt_ld := nvinfer nvonnxparser
tensorrt_ld := nvinfer nvinfer_plugin
#sys_ld      := stdc++ dl
sys_ld      := stdc++ dl protobuf
serialport  := CppLinuxSerial
ld_librarys := $(cuda_ld) $(opencv_ld) $(sys_ld) $(tensorrt_ld) $(serialport)
ld_librarys := $(ld_librarys:%=-l%)


cpp_compile_flags := -std=c++11 -w -g -O0 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=c++11 -w -O0 -Xcompiler "$(cpp_compile_flags)"
link_flags := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags += $(library_paths) $(ld_librarys) $(run_paths)

#头文件修改后策略
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mks) $(cu_mks)
endif

# .mk生成
# @g++ -MM $< -MF $@ -MT $(@:.mk=.o)
# 编译cpp依赖项，生成mk文件
objs/%.cpp.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@g++ -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
objs/%.cu.mk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

#编译CXX
objs/%.cpp.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(cpp_compile_flags)

#编译CUDA
objs/%.cu.o : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

#链接
workspace/pro : $(cpp_objs) $(cu_objs)
	@printf "\n$(RED_COLOR)开始链接至$@...$(NO_COLOR)\n"
	@mkdir -p $(dir $@)
	g++ $^ -o $@ $(link_flags)

pro : workspace/pro
#@echo 编译完成 生成可执行文件$@
	@printf "\n$(RED_COLOR)编译完成，生成可执行文件$<$(NO_COLOR)\n"

run : pro
	@printf "\n$(RED_COLOR)成功运行...$(NO_COLOR)\n"
	@cd workspace && ./pro

clean:
	rm -rf workspace/pro objs

debug : workspace/pro
	@printf "\n$(RED_COLOR)readelf...$(NO_COLOR)\n"
	@readelf -d $<
	@printf "\n$(RED_COLOR)ldd...$(NO_COLOR)\n"
	@ldd $<

again:
	@make clean && make run

.PHONY: pro run clean debug again

var:
	@echo $(library_path_export)

# 导出依赖库路径，使得能够运行起来

export LD_LIBRARY_PATH:=$(library_path_export)

