#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace plapoint
{

    /// Base exception with the source-location accessors exposed by PCL.
    class PCLException : public std::runtime_error
    {
    public:
        PCLException(const std::string& description,
                     const char* file_name = nullptr,
                     const char* function_name = nullptr,
                     unsigned line_number = 0)
            : std::runtime_error(createDetailedMessage(description, file_name, function_name, line_number)),
              file_name_(file_name), function_name_(function_name), line_number_(line_number)
        {
        }

        const char* getFileName() const noexcept
        {
            return file_name_;
        }
        const char* getFunctionName() const noexcept
        {
            return function_name_;
        }
        unsigned getLineNumber() const noexcept
        {
            return line_number_;
        }
        const char* detailedMessage() const noexcept
        {
            return what();
        }

    protected:
        static std::string createDetailedMessage(const std::string& description,
                                                 const char* file_name,
                                                 const char* function_name,
                                                 unsigned line_number)
        {
            std::ostringstream output;
            if (function_name)
            {
                output << function_name << ' ';
            }
            if (file_name)
            {
                output << "in " << file_name << ' ';
                if (line_number != 0)
                {
                    output << "@ " << line_number << ' ';
                }
            }
            output << ": " << description;
            return output.str();
        }

        const char* file_name_;
        const char* function_name_;
        unsigned line_number_;
    };

    /// Raised when a two-dimensional accessor is used on an unorganized cloud.
    class UnorganizedPointCloudException : public PCLException
    {
    public:
        using PCLException::PCLException;
    };

} // namespace plapoint
