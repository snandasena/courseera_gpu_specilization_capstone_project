# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)

all: clean build

build: src/sobel_edge_detection.cu src/kernels.h
	$(CXX) --std c++17  -I/usr/include/opencv4 -I/usr/local/cuda/include \
	 src/sobel_edge_detection.cu  -o bin/sobel_edge_detection.exe \
	`pkg-config opencv4 --cflags --libs` -lcuda -lstdc++ \
	-ccbin /usr/bin/gcc-10 \
	-Wno-deprecated-gpu-targets


run:
	./bin/sobel_edge_detection.exe $(ARGS) > logs/outputs.log

clean:
	rm -rf bin/* logs/*