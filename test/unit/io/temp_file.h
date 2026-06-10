#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace plapoint::test
{

class TempFile
{
public:
    explicit TempFile(const std::string& suffix, const std::string& prefix = "plapoint_test")
        : _path(makePath(prefix, suffix))
    {
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    TempFile(TempFile&&) = delete;
    TempFile& operator=(TempFile&&) = delete;

    ~TempFile()
    {
        std::error_code ignored;
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

    std::string string() const
    {
        return _path.string();
    }

private:
    static std::filesystem::path makePath(const std::string& prefix, const std::string& suffix)
    {
        static std::atomic<unsigned long long> counter{0};
        std::random_device random;

        for (int attempt = 0; attempt < 16; ++attempt)
        {
            const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            const auto serial = counter.fetch_add(1, std::memory_order_relaxed);
            std::ostringstream name;
            name << prefix << '_' << now << '_' << random() << '_' << serial << suffix;

            auto candidate = std::filesystem::temp_directory_path() / name.str();
            if (!std::filesystem::exists(candidate))
            {
                return candidate;
            }
        }

        throw std::runtime_error("TempFile: failed to create a unique temporary path");
    }

    std::filesystem::path _path;
};

} // namespace plapoint::test
