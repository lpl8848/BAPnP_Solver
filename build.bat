@echo off
call "D:\CPP\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d D:\BAPnP_Solver
echo ===== Configuring CMake =====
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -G Ninja
if %ERRORLEVEL% NEQ 0 (
    echo ===== CMake configure FAILED =====
    exit /b 1
)
echo ===== Building =====
cmake --build build --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ===== Build FAILED =====
    exit /b 1
)
echo ===== BUILD SUCCESS =====
dir build\*.exe
