cd src
rm -rf build
mkdir build
python3 gen2.py pycld build headers.txt
g++ -shared -rdynamic -g -O3 -Wall -fPIC \
        cld.cpp \
        src/ETF.cpp \
        src/FDoG.cpp \
        -DMODULE_STR=cld -DMODULE_PREFIX=pycld \
        -DNDEBUG -DPY_MAJOR_VERSION=3 \
        `pkg-config --cflags --libs opencv`  \
        `python3-config --includes --ldflags` \
        -I . -I/usr/local/lib/python3.7/site-packages/numpy/core/include \
        -o ../cld.so 
rm -rf build
cd ..