#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint
{

template <typename PointT> class PointRepresentation
{
public:
    using Ptr = std::shared_ptr<PointRepresentation<PointT>>;
    using ConstPtr = std::shared_ptr<const PointRepresentation<PointT>>;

    virtual ~PointRepresentation() = default;
    virtual void copyToFloatArray(const PointT& point, float* output) const = 0;

    bool isTrivial() const
    {
        return trivial_ && alpha_.empty();
    }

    virtual bool isValid(const PointT& point) const
    {
        std::vector<float> values(static_cast<std::size_t>(nr_dimensions_));
        copyToFloatArray(point, values.data());
        return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    }

    template <typename OutputType> void vectorize(const PointT& point, OutputType& output) const
    {
        std::vector<float> values(static_cast<std::size_t>(nr_dimensions_));
        copyToFloatArray(point, values.data());
        for (int dimension = 0; dimension < nr_dimensions_; ++dimension)
        {
            output[dimension] = alpha_.empty() ? values[dimension] : values[dimension] * alpha_[dimension];
        }
    }

    void vectorize(const PointT& point, float* output) const
    {
        copyToFloatArray(point, output);
        if (!alpha_.empty())
        {
            for (int dimension = 0; dimension < nr_dimensions_; ++dimension)
            {
                output[dimension] *= alpha_[dimension];
            }
        }
    }

    void vectorize(const PointT& point, std::vector<float>& output) const
    {
        if (output.size() < static_cast<std::size_t>(nr_dimensions_))
        {
            output.resize(static_cast<std::size_t>(nr_dimensions_));
        }
        vectorize(point, output.data());
    }

    void setRescaleValues(const float* values)
    {
        if (!values)
        {
            throw std::invalid_argument("PointRepresentation: rescale values must not be null");
        }
        alpha_.assign(values, values + nr_dimensions_);
    }

    int getNumberOfDimensions() const
    {
        return nr_dimensions_;
    }

protected:
    int nr_dimensions_ = 0;
    std::vector<float> alpha_;
    bool trivial_ = false;
};

template <typename PointT> class DefaultPointRepresentation : public PointRepresentation<PointT>
{
public:
    using Ptr = std::shared_ptr<DefaultPointRepresentation<PointT>>;
    using ConstPtr = std::shared_ptr<const DefaultPointRepresentation<PointT>>;

    DefaultPointRepresentation()
    {
        this->nr_dimensions_ = std::min<int>(3, static_cast<int>(sizeof(PointT) / sizeof(float)));
        this->trivial_ = true;
    }

    Ptr makeShared() const
    {
        return std::make_shared<DefaultPointRepresentation<PointT>>(*this);
    }

    void copyToFloatArray(const PointT& point, float* output) const override
    {
        const float* values = reinterpret_cast<const float*>(&point);
        std::copy(values, values + this->nr_dimensions_, output);
    }
};

} // namespace plapoint
