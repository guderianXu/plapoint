#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

namespace plapoint
{
namespace search
{

template <std::size_t Dimension, typename Scalar = double>
struct SpatialPoint
{
    static_assert(Dimension > 0, "SpatialPoint Dimension must be greater than zero");
    static_assert(std::is_arithmetic<Scalar>::value, "SpatialPoint Scalar must be arithmetic");

    std::array<Scalar, Dimension> coords{};
    int index = -1;
};

template <typename Scalar>
struct SpatialNeighbor
{
    int index = -1;
    Scalar distanceSquared = Scalar(0);
};

template <std::size_t Dimension, typename Scalar = double>
class SpatialKdTree
{
public:
    static_assert(Dimension > 0, "SpatialKdTree Dimension must be greater than zero");
    static_assert(std::is_arithmetic<Scalar>::value, "SpatialKdTree Scalar must be arithmetic");

    using Point = SpatialPoint<Dimension, Scalar>;
    using CoordinateArray = std::array<Scalar, Dimension>;
    using Neighbor = SpatialNeighbor<Scalar>;

public:
    SpatialKdTree() = default;

    explicit SpatialKdTree(const std::vector<Point>& points)
    {
        build(points);
    }

    explicit SpatialKdTree(const std::vector<CoordinateArray>& coords)
    {
        build(coords);
    }

    void build(const std::vector<Point>& points)
    {
        _points = points;
        rebuild();
    }

    void build(const std::vector<CoordinateArray>& coords)
    {
        _points.clear();
        _points.reserve(coords.size());

        assert(coords.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
               "SpatialKdTree: point count exceeds int range");
        for (std::size_t index = 0; index < coords.size(); ++index)
        {
            Point point;
            point.coords = coords[index];
            point.index = static_cast<int>(index);
            _points.push_back(point);
        }

        rebuild();
    }

    void clear()
    {
        _points.clear();
        _nodes.clear();
        _root = -1;
    }

    bool empty() const
    {
        return _points.empty();
    }

    std::size_t size() const
    {
        return _points.size();
    }

    const std::vector<Point>& points() const
    {
        return _points;
    }

    int nearest(const CoordinateArray& query, Scalar* outDistance = nullptr) const
    {
        if (_root < 0)
        {
            return -1;
        }

        int bestPoint = -1;
        Scalar bestDistSq = std::numeric_limits<Scalar>::max();
        nearestRecursive(_root, query, -1, bestPoint, bestDistSq);

        if (outDistance)
        {
            *outDistance = std::sqrt(bestDistSq);
        }

        return bestPoint < 0 ? -1 : _points[static_cast<std::size_t>(bestPoint)].index;
    }

    int nearestByPointIndex(std::size_t pointIndex, Scalar* outDistance = nullptr) const
    {
        if (pointIndex >= _points.size())
        {
            return -1;
        }

        return nearestSkipping(_points[pointIndex].coords, static_cast<int>(pointIndex), outDistance);
    }

    std::vector<Neighbor> kNearest(const CoordinateArray& query, std::size_t k) const
    {
        return kNearestSkipping(query, k, -1);
    }

    std::vector<Neighbor> kNearestByPointIndex(std::size_t pointIndex, std::size_t k) const
    {
        if (pointIndex >= _points.size())
        {
            return {};
        }

        return kNearestSkipping(_points[pointIndex].coords, k, static_cast<int>(pointIndex));
    }

    std::vector<int> radiusSearch(const CoordinateArray& query, Scalar radius) const
    {
        std::vector<int> result;
        radiusSearchInternal(query, radius, -1, &result, nullptr, std::numeric_limits<int>::max());
        return result;
    }

    std::vector<int> radiusSearchByPointIndex(std::size_t pointIndex, Scalar radius) const
    {
        if (pointIndex >= _points.size())
        {
            return {};
        }

        std::vector<int> result;
        radiusSearchInternal(_points[pointIndex].coords,
                             radius,
                             static_cast<int>(pointIndex),
                             &result,
                             nullptr,
                             std::numeric_limits<int>::max());
        return result;
    }

    int radiusCount(const CoordinateArray& query, Scalar radius, int earlyStop = std::numeric_limits<int>::max()) const
    {
        int count = 0;
        radiusSearchInternal(query, radius, -1, nullptr, &count, earlyStop);
        return count;
    }

    int radiusCountByPointIndex(std::size_t pointIndex,
                                Scalar radius,
                                int earlyStop = std::numeric_limits<int>::max()) const
    {
        if (pointIndex >= _points.size())
        {
            return 0;
        }

        int count = 0;
        radiusSearchInternal(_points[pointIndex].coords,
                             radius,
                             static_cast<int>(pointIndex),
                             nullptr,
                             &count,
                             earlyStop);
        return count;
    }

private:
    struct Node
    {
        int point = -1;
        int axis = 0;
        int left = -1;
        int right = -1;
    };

    struct HeapEntry
    {
        Scalar distanceSquared = Scalar(0);
        int point = -1;

        bool operator<(const HeapEntry& other) const
        {
            return distanceSquared < other.distanceSquared;
        }
    };

private:
    void rebuild()
    {
        _nodes.clear();
        _root = -1;
        if (_points.empty())
        {
            return;
        }

        std::vector<int> indices(_points.size());
        std::iota(indices.begin(), indices.end(), 0);
        _nodes.reserve(_points.size());
        _root = buildRecursive(indices, 0, static_cast<int>(indices.size()));
    }

    int buildRecursive(std::vector<int>& indices, int begin, int end)
    {
        if (begin >= end)
        {
            return -1;
        }

        const int axis = selectSplitAxis(indices, begin, end);
        const int mid = (begin + end) / 2;

        std::nth_element(indices.begin() + begin,
                         indices.begin() + mid,
                         indices.begin() + end,
                         [this, axis](int lhs, int rhs) {
                             return _points[static_cast<std::size_t>(lhs)].coords[static_cast<std::size_t>(axis)] <
                                    _points[static_cast<std::size_t>(rhs)].coords[static_cast<std::size_t>(axis)];
                         });

        Node node;
        node.point = indices[mid];
        node.axis = axis;

        const int nodeIndex = static_cast<int>(_nodes.size());
        _nodes.push_back(node);
        _nodes[static_cast<std::size_t>(nodeIndex)].left = buildRecursive(indices, begin, mid);
        _nodes[static_cast<std::size_t>(nodeIndex)].right = buildRecursive(indices, mid + 1, end);
        return nodeIndex;
    }

    int selectSplitAxis(const std::vector<int>& indices, int begin, int end) const
    {
        CoordinateArray minValues{};
        CoordinateArray maxValues{};
        minValues.fill(std::numeric_limits<Scalar>::max());
        maxValues.fill(std::numeric_limits<Scalar>::lowest());

        for (int i = begin; i < end; ++i)
        {
            const Point& point = _points[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])];
            for (std::size_t axis = 0; axis < Dimension; ++axis)
            {
                minValues[axis] = std::min(minValues[axis], point.coords[axis]);
                maxValues[axis] = std::max(maxValues[axis], point.coords[axis]);
            }
        }

        int bestAxis = 0;
        Scalar bestExtent = maxValues[0] - minValues[0];
        for (std::size_t axis = 1; axis < Dimension; ++axis)
        {
            const Scalar extent = maxValues[axis] - minValues[axis];
            if (extent > bestExtent)
            {
                bestExtent = extent;
                bestAxis = static_cast<int>(axis);
            }
        }

        return bestAxis;
    }

    static Scalar distanceSquared(const CoordinateArray& lhs, const CoordinateArray& rhs)
    {
        Scalar result = Scalar(0);
        for (std::size_t axis = 0; axis < Dimension; ++axis)
        {
            const Scalar delta = lhs[axis] - rhs[axis];
            result += delta * delta;
        }
        return result;
    }

    int nearestSkipping(const CoordinateArray& query, int skipPoint, Scalar* outDistance) const
    {
        if (_root < 0)
        {
            return -1;
        }

        int bestPoint = -1;
        Scalar bestDistSq = std::numeric_limits<Scalar>::max();
        nearestRecursive(_root, query, skipPoint, bestPoint, bestDistSq);

        if (outDistance)
        {
            *outDistance = bestPoint >= 0 ? std::sqrt(bestDistSq) : Scalar(0);
        }

        return bestPoint < 0 ? -1 : _points[static_cast<std::size_t>(bestPoint)].index;
    }

    void nearestRecursive(int nodeIndex,
                          const CoordinateArray& query,
                          int skipPoint,
                          int& bestPoint,
                          Scalar& bestDistSq) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        const Node& node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point& point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (currentDistSq < bestDistSq)
            {
                bestDistSq = currentDistSq;
                bestPoint = node.point;
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        nearestRecursive(nearChild, query, skipPoint, bestPoint, bestDistSq);

        if (delta * delta < bestDistSq)
        {
            nearestRecursive(farChild, query, skipPoint, bestPoint, bestDistSq);
        }
    }

    std::vector<Neighbor> kNearestSkipping(const CoordinateArray& query, std::size_t k, int skipPoint) const
    {
        if (_root < 0 || k == 0)
        {
            return {};
        }

        std::priority_queue<HeapEntry> heap;
        kNearestRecursive(_root, query, k, skipPoint, heap);

        std::vector<Neighbor> result(heap.size());
        for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i)
        {
            const HeapEntry entry = heap.top();
            heap.pop();
            result[static_cast<std::size_t>(i)] = Neighbor{
                _points[static_cast<std::size_t>(entry.point)].index,
                entry.distanceSquared
            };
        }
        return result;
    }

    void kNearestRecursive(int nodeIndex,
                           const CoordinateArray& query,
                           std::size_t k,
                           int skipPoint,
                           std::priority_queue<HeapEntry>& heap) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        const Node& node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point& point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (heap.size() < k)
            {
                heap.push(HeapEntry{currentDistSq, node.point});
            }
            else if (currentDistSq < heap.top().distanceSquared)
            {
                heap.pop();
                heap.push(HeapEntry{currentDistSq, node.point});
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        kNearestRecursive(nearChild, query, k, skipPoint, heap);

        if (heap.size() < k || delta * delta < heap.top().distanceSquared)
        {
            kNearestRecursive(farChild, query, k, skipPoint, heap);
        }
    }

    void radiusSearchInternal(const CoordinateArray& query,
                              Scalar radius,
                              int skipPoint,
                              std::vector<int>* results,
                              int* count,
                              int earlyStop) const
    {
        if (_root < 0 || radius <= Scalar(0))
        {
            return;
        }

        const Scalar radiusSq = radius * radius;
        radiusRecursive(_root, query, radiusSq, skipPoint, results, count, earlyStop);
    }

    void radiusRecursive(int nodeIndex,
                         const CoordinateArray& query,
                         Scalar radiusSq,
                         int skipPoint,
                         std::vector<int>* results,
                         int* count,
                         int earlyStop) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        if (count && *count >= earlyStop)
        {
            return;
        }

        const Node& node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point& point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (currentDistSq <= radiusSq)
            {
                if (results)
                {
                    results->push_back(point.index);
                }

                if (count)
                {
                    ++(*count);
                    if (*count >= earlyStop)
                    {
                        return;
                    }
                }
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        radiusRecursive(nearChild, query, radiusSq, skipPoint, results, count, earlyStop);

        if ((!count || *count < earlyStop) && delta * delta <= radiusSq)
        {
            radiusRecursive(farChild, query, radiusSq, skipPoint, results, count, earlyStop);
        }
    }

private:
    std::vector<Point> _points;
    std::vector<Node> _nodes;
    int _root = -1;
};

using SpatialKdTree2Dd = SpatialKdTree<2, double>;
using SpatialKdTree2Df = SpatialKdTree<2, float>;
using SpatialKdTree3Dd = SpatialKdTree<3, double>;
using SpatialKdTree3Df = SpatialKdTree<3, float>;

} // namespace search
} // namespace plapoint
