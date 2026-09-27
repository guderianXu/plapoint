#pragma once

#include <memory>

namespace plapoint::registration
{

    class ConvergenceCriteria
    {
    public:
        using Ptr = std::shared_ptr<ConvergenceCriteria>;
        using ConstPtr = std::shared_ptr<const ConvergenceCriteria>;

        ConvergenceCriteria() = default;
        virtual ~ConvergenceCriteria() = default;

        virtual bool hasConverged() = 0;

        operator bool()
        {
            return hasConverged();
        }
    };

} // namespace plapoint::registration
