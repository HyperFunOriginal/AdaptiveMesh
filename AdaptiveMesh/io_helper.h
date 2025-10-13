#ifndef IO_HELPER_H
#define IO_HELPER_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "CUDA_memory.h"

template <class T>
void write_to_file(const smart_cpu_buffer<T>& arr, const char* filepath)
{
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
void write_to_file(smart_gpu_cpu_buffer<T>& arr, const char* filepath, const bool copy_to_cpu_first = false)
{
    if (copy_to_cpu_first) { arr.copy_to_cpu(); }
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
smart_cpu_buffer<T> read_from_file(const char* filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    file.seekg(0, std::ios::end);
    size_t len = file.tellg();
    file.seekg(0, file.beg);

    smart_cpu_buffer<T> result(len / sizeof(T));
    char* ptr = reinterpret_cast<char*>(result.cpu_buffer_ptr);
    file.read(ptr, min(len, result.dedicated_len * sizeof(T)));
    return result;
}

#include <filesystem>
#pragma message("NOTE: -std=c++17 -Xcompiler \"/std:c++17\" CUDA command required for compilation.")
using namespace std::filesystem;

bool create_folder(const char* path)
{
    try {
        std::filesystem::create_directory(path);
        return true;
    }
    catch (...)
    {
        return false;
    }
}
/// <summary>
/// Checks if folder at the specified path exists and returns true if it does. If not, it creates the directory returning false.
/// </summary>
/// <param name="path">Filepath to the directory</param>
/// <returns></returns>
bool check_folder_exists_create(const char* path)
{
    return !std::filesystem::create_directory(path);
}
std::vector<std::string> read_all_filepaths_in_dir(const char* dir)
{
    std::vector<std::string> paths;
    for (std::filesystem::directory_iterator iter(dir); !iter._At_end(); iter++)
    {
        std::filesystem::directory_entry entry = (*iter);
        if (entry.is_regular_file())
            paths.push_back(entry.path().string());
    }
    return paths;
}
template <class... Args>
void apply_to_all_files_in_dir(const char* dir, void (*action)(const std::filesystem::path& path, Args... args), Args... args)
{
    for (std::filesystem::directory_iterator iter(dir); !iter._At_end(); iter++)
    {
        std::filesystem::directory_entry entry = (*iter);
        if (entry.is_regular_file())
            action(entry.path(), args...);
    }
}

#endif