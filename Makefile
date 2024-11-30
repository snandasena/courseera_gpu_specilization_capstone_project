# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)

all: clean build

build: src/main.cu src/kernels.h
	$(CXX)  src/main.cu --std c++17 `pkg-config opencv4 --cflags --libs` -o  bin/main.exe -Wno-deprecated-gpu-targets -I/usr/include/opencv4 -I/usr/local/cuda/include -lcuda -lstdc++ -ccbin /usr/bin/gcc-10


run:
	./bin/main.exe $(ARGS) > logs/outputs.log

clean:
	rm -rf bin/* logs/*