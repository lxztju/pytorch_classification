if [ ! -d "./build/" ];then
    mkdir build 
fi
cd build &&cmake -DCMAKE_CXX_STANDARD=14 .. && make -j2