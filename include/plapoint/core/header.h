#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

namespace plapoint
{

    /// Acquisition metadata carried by PCL point clouds.
    struct PCLHeader
    {
        using Ptr = std::shared_ptr<PCLHeader>;
        using ConstPtr = std::shared_ptr<const PCLHeader>;

        std::uint32_t seq = 0;
        std::uint64_t stamp = 0;
        std::string frame_id;
    };

    using HeaderPtr = PCLHeader::Ptr;
    using HeaderConstPtr = PCLHeader::ConstPtr;

    inline bool operator==(const PCLHeader& lhs, const PCLHeader& rhs)
    {
        return &lhs == &rhs || (lhs.seq == rhs.seq && lhs.stamp == rhs.stamp && lhs.frame_id == rhs.frame_id);
    }

    inline std::ostream& operator<<(std::ostream& output, const PCLHeader& header)
    {
        return output << "seq: " << header.seq << " stamp: " << header.stamp << " frame_id: " << header.frame_id
                      << std::endl;
    }

} // namespace plapoint
