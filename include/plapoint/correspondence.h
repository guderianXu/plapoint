#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <ostream>
#include <unordered_set>
#include <vector>

#include <Eigen/StdVector>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{

inline constexpr index_t UNAVAILABLE = -1;

struct Correspondence
{
    index_t index_query = 0;
    index_t index_match = UNAVAILABLE;
    union
    {
        float distance = std::numeric_limits<float>::max();
        float weight;
    };

    Correspondence() = default;
    Correspondence(index_t query, index_t match, float correspondence_distance)
        : index_query(query), index_match(match), distance(correspondence_distance)
    {
    }
};

inline std::ostream& operator<<(std::ostream& stream, const Correspondence& correspondence)
{
    return stream << correspondence.index_query << ' ' << correspondence.index_match << ' '
                  << correspondence.distance;
}

using Correspondences = std::vector<Correspondence, Eigen::aligned_allocator<Correspondence>>;
using CorrespondencesPtr = std::shared_ptr<Correspondences>;
using CorrespondencesConstPtr = std::shared_ptr<const Correspondences>;

inline void getRejectedQueryIndices(const Correspondences& before, const Correspondences& after,
                                    Indices& indices, bool presorting_required = true)
{
    indices.clear();
    if (presorting_required)
    {
        std::unordered_set<index_t> kept;
        kept.reserve(after.size());
        for (const auto& correspondence : after)
        {
            kept.insert(correspondence.index_query);
        }
        for (const auto& correspondence : before)
        {
            if (kept.find(correspondence.index_query) == kept.end())
            {
                indices.push_back(correspondence.index_query);
            }
        }
        return;
    }

    auto after_iterator = after.begin();
    for (const auto& correspondence : before)
    {
        if (after_iterator != after.end() &&
            after_iterator->index_query == correspondence.index_query)
        {
            ++after_iterator;
        }
        else
        {
            indices.push_back(correspondence.index_query);
        }
    }
}

inline bool isBetterCorrespondence(const Correspondence& lhs, const Correspondence& rhs)
{
    return lhs.distance > rhs.distance;
}

} // namespace plapoint
