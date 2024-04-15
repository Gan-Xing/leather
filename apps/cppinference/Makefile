CXX=g++
CXXFLAGS=-std=c++17 -O2 `pkg-config --cflags opencv4`
LDFLAGS=`pkg-config --libs opencv4` -lonnxruntime

TARGET=run_inference

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
