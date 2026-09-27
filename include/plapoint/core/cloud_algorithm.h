#pragma once

#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{

template <typename PointT> class PCLBase
{
public:
    using PointCloud = plapoint::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;
    using PointIndicesPtr = PointIndices::Ptr;
    using PointIndicesConstPtr = PointIndices::ConstPtr;

    PCLBase() = default;
    PCLBase(const PCLBase&) = default;
    virtual ~PCLBase() = default;

    virtual void setInputCloud(const PointCloudConstPtr& cloud)
    {
        if (input_ != cloud)
        {
            input_ = cloud;
            if (fake_indices_ && (!input_ || indices_->size() != input_->size()))
            {
                indices_.reset();
                fake_indices_ = false;
            }
        }
    }

    PointCloudConstPtr getInputCloud() const
    {
        return input_;
    }

    virtual void setIndices(const IndicesPtr& indices)
    {
        indices_ = indices;
        use_indices_ = static_cast<bool>(indices);
        fake_indices_ = false;
    }

    virtual void setIndices(const IndicesConstPtr& indices)
    {
        indices_ = indices ? std::make_shared<Indices>(*indices) : IndicesPtr{};
        use_indices_ = static_cast<bool>(indices);
        fake_indices_ = false;
    }

    virtual void setIndices(const PointIndicesConstPtr& indices)
    {
        setIndices(indices ? std::make_shared<Indices>(indices->indices) : IndicesPtr{});
    }

    virtual void setIndices(std::size_t row_start, std::size_t col_start,
                            std::size_t nb_rows, std::size_t nb_cols)
    {
        if (!input_ || !input_->isOrganized())
        {
            throw std::invalid_argument("PCLBase: region indices require an organized input cloud");
        }
        if (row_start > input_->height || col_start > input_->width ||
            nb_rows > input_->height - row_start || nb_cols > input_->width - col_start)
        {
            throw std::out_of_range("PCLBase: region lies outside the input cloud");
        }
        auto indices = std::make_shared<Indices>();
        indices->reserve(nb_rows * nb_cols);
        for (std::size_t row = row_start; row < row_start + nb_rows; ++row)
        {
            for (std::size_t column = col_start; column < col_start + nb_cols; ++column)
            {
                const std::size_t index = row * input_->width + column;
                if (index > static_cast<std::size_t>(std::numeric_limits<index_t>::max()))
                {
                    throw std::overflow_error("PCLBase: point index exceeds index_t range");
                }
                indices->push_back(static_cast<index_t>(index));
            }
        }
        setIndices(indices);
    }

    IndicesPtr getIndices()
    {
        return indices_;
    }

    IndicesConstPtr getIndices() const
    {
        return indices_;
    }

    const PointT& operator[](std::size_t position) const
    {
        return (*input_)[static_cast<std::size_t>((*indices_)[position])];
    }

protected:
    bool initCompute()
    {
        if (!input_)
        {
            return false;
        }
        if (!indices_)
        {
            indices_ = std::make_shared<Indices>(input_->size());
            std::iota(indices_->begin(), indices_->end(), 0);
            fake_indices_ = true;
        }
        return true;
    }

    bool deinitCompute()
    {
        return true;
    }

    PointCloudConstPtr input_;
    IndicesPtr indices_;
    bool use_indices_ = false;
    bool fake_indices_ = false;
};

} // namespace plapoint
