#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/filter.h>

namespace plapoint
{
    namespace detail
    {

        /// Shared PCL container and index behavior for the matrix-backed outlier filters.
        template <typename PointT, typename FilterT> class PointOutlierAdapter : public FilterIndices<PointT>
        {
        public:
            using PointCloudType = plapoint::PointCloud<PointT>;
            using PointCloudConstPtr = typename PointCloudType::ConstPtr;
            using Ptr = std::shared_ptr<FilterT>;
            using ConstPtr = std::shared_ptr<const FilterT>;

            explicit PointOutlierAdapter(bool extract_removed_indices = false)
                : FilterIndices<PointT>(extract_removed_indices)
            {
            }

        protected:
            PointCloudConstPtr inputCloud() const
            {
                if (!this->input_)
                {
                    throw std::runtime_error("OutlierRemoval: input cloud not set");
                }
                return this->input_;
            }

            IndicesConstPtr inputIndices() const noexcept
            {
                return this->use_indices_ ? this->indices_ : IndicesConstPtr{};
            }

            void applyFilter(Indices& indices) override
            {
                const auto selection = selectIndices();
                indices = selection.kept;
            }

        private:
            struct Selection
            {
                Indices kept;
                Indices removed;
            };

            Selection selectIndices()
            {
                inputCloud();
                const auto rejected = static_cast<FilterT*>(this)->computeRejectedIndices();
                std::vector<bool> rejected_mask(this->input_->size(), false);
                for (const int index : rejected)
                {
                    if (index < 0 || static_cast<std::size_t>(index) >= this->input_->size())
                    {
                        throw std::out_of_range("OutlierRemoval: rejected index is outside the input cloud");
                    }
                    rejected_mask[static_cast<std::size_t>(index)] = true;
                }

                Selection selection;
                const auto selected = inputIndices();
                const std::size_t count = selected ? selected->size() : this->input_->size();
                selection.kept.reserve(count);
                selection.removed.reserve(count);
                for (std::size_t row = 0; row < count; ++row)
                {
                    const int index = selected ? selected->at(row) : static_cast<int>(row);
                    if (index < 0 || static_cast<std::size_t>(index) >= this->input_->size())
                    {
                        throw std::out_of_range("OutlierRemoval: input index is outside the cloud");
                    }
                    const bool keep = rejected_mask[static_cast<std::size_t>(index)] == this->negative_;
                    (keep ? selection.kept : selection.removed).push_back(index);
                }
                *this->removed_indices_ = this->extract_removed_indices_ ? selection.removed : Indices{};
                return selection;
            }
        };

    } // namespace detail
} // namespace plapoint
