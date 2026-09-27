#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <plapoint/core/header.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/core/point_traits.h>

namespace plapoint
{

    struct PCLPointField
    {
        enum PointFieldTypes : std::uint8_t
        {
            INT8 = 1,
            UINT8 = 2,
            INT16 = 3,
            UINT16 = 4,
            INT32 = 5,
            UINT32 = 6,
            FLOAT32 = 7,
            FLOAT64 = 8,
            INT64 = 9,
            UINT64 = 10,
            BOOL = 11
        };

        using Ptr = std::shared_ptr<PCLPointField>;
        using ConstPtr = std::shared_ptr<const PCLPointField>;

        std::string name;
        uindex_t offset = 0;
        std::uint8_t datatype = 0;
        uindex_t count = 0;
    };

    using PCLPointFieldPtr = PCLPointField::Ptr;
    using PCLPointFieldConstPtr = PCLPointField::ConstPtr;

    inline std::size_t getFieldSize(int datatype)
    {
        switch (datatype)
        {
        case PCLPointField::INT8:
        case PCLPointField::UINT8:
        case PCLPointField::BOOL:
            return 1;
        case PCLPointField::INT16:
        case PCLPointField::UINT16:
            return 2;
        case PCLPointField::INT32:
        case PCLPointField::UINT32:
        case PCLPointField::FLOAT32:
            return 4;
        case PCLPointField::FLOAT64:
        case PCLPointField::INT64:
        case PCLPointField::UINT64:
            return 8;
        default:
            return 0;
        }
    }

    namespace detail
    {

        inline long double readNumericField(const std::uint8_t* data, std::uint8_t datatype)
        {
            switch (datatype)
            {
#define PLAPOINT_READ_BLOB_NUMBER(code, type)                                                                          \
    case code:                                                                                                         \
    {                                                                                                                  \
        type value{};                                                                                                  \
        std::memcpy(&value, data, sizeof(value));                                                                      \
        return static_cast<long double>(value);                                                                        \
    }
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::INT8, std::int8_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::UINT8, std::uint8_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::INT16, std::int16_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::UINT16, std::uint16_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::INT32, std::int32_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::UINT32, std::uint32_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::FLOAT32, float)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::FLOAT64, double)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::INT64, std::int64_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::UINT64, std::uint64_t)
                PLAPOINT_READ_BLOB_NUMBER(PCLPointField::BOOL, bool)
#undef PLAPOINT_READ_BLOB_NUMBER
            default:
                throw std::invalid_argument("fromPCLPointCloud2: unsupported field datatype");
            }
        }

        inline bool hostIsBigEndian() noexcept
        {
            const std::uint16_t value = 0x0102;
            return *reinterpret_cast<const std::uint8_t*>(&value) == 0x01;
        }

        inline bool equivalentFieldName(const std::string& lhs, const std::string& rhs)
        {
            return lhs == rhs || ((lhs == "rgb" || lhs == "rgba") && (rhs == "rgb" || rhs == "rgba"));
        }

    } // namespace detail

    struct PCLPointCloud2
    {
        using Ptr = std::shared_ptr<PCLPointCloud2>;
        using ConstPtr = std::shared_ptr<const PCLPointCloud2>;

        PCLHeader header;
        uindex_t height = 0;
        uindex_t width = 0;
        std::vector<PCLPointField> fields;
        std::uint8_t is_bigendian = static_cast<std::uint8_t>(detail::hostIsBigEndian());
        uindex_t point_step = 0;
        uindex_t row_step = 0;
        std::vector<std::uint8_t> data;
        std::uint8_t is_dense = 0;

        static bool concatenate(PCLPointCloud2& lhs, const PCLPointCloud2& rhs)
        {
            if (lhs.is_bigendian != rhs.is_bigendian || lhs.point_step != rhs.point_step ||
                lhs.fields.size() != rhs.fields.size())
            {
                return false;
            }
            for (std::size_t index = 0; index < lhs.fields.size(); ++index)
            {
                const auto& a = lhs.fields[index];
                const auto& b = rhs.fields[index];
                if (!detail::equivalentFieldName(a.name, b.name) || a.offset != b.offset || a.datatype != b.datatype ||
                    a.count != b.count)
                {
                    return false;
                }
            }
            const std::uint64_t lhs_points = static_cast<std::uint64_t>(lhs.width) * lhs.height;
            const std::uint64_t rhs_points = static_cast<std::uint64_t>(rhs.width) * rhs.height;
            if (lhs_points + rhs_points > std::numeric_limits<uindex_t>::max())
            {
                return false;
            }
            lhs.data.insert(lhs.data.end(), rhs.data.begin(), rhs.data.end());
            lhs.width = static_cast<uindex_t>(lhs_points + rhs_points);
            lhs.height = lhs.width == 0 ? 0u : 1u;
            lhs.row_step = lhs.width * lhs.point_step;
            lhs.is_dense = static_cast<std::uint8_t>(lhs.is_dense && rhs.is_dense);
            lhs.header.stamp = std::max(lhs.header.stamp, rhs.header.stamp);
            return true;
        }

        static bool concatenate(const PCLPointCloud2& lhs, const PCLPointCloud2& rhs, PCLPointCloud2& output)
        {
            output = lhs;
            return concatenate(output, rhs);
        }

        PCLPointCloud2& operator+=(const PCLPointCloud2& rhs)
        {
            concatenate(*this, rhs);
            return *this;
        }

        PCLPointCloud2 operator+(const PCLPointCloud2& rhs)
        {
            PCLPointCloud2 output(*this);
            output += rhs;
            return output;
        }

        template <typename T> const T& at(const uindex_t& point_index, const uindex_t& field_offset) const
        {
            return atImpl<T>(point_index, field_offset);
        }

        template <typename T> T& at(const uindex_t& point_index, const uindex_t& field_offset)
        {
            return atImpl<T>(point_index, field_offset);
        }

    private:
        template <typename T> T& atImpl(uindex_t point_index, uindex_t field_offset)
        {
            const std::uint64_t position = static_cast<std::uint64_t>(point_index) * point_step + field_offset;
            if (position + sizeof(T) > data.size())
            {
                throw std::out_of_range("PCLPointCloud2::at");
            }
            return *reinterpret_cast<T*>(data.data() + position);
        }

        template <typename T> const T& atImpl(uindex_t point_index, uindex_t field_offset) const
        {
            const std::uint64_t position = static_cast<std::uint64_t>(point_index) * point_step + field_offset;
            if (position + sizeof(T) > data.size())
            {
                throw std::out_of_range("PCLPointCloud2::at");
            }
            return *reinterpret_cast<const T*>(data.data() + position);
        }
    };

    using PCLPointCloud2Ptr = PCLPointCloud2::Ptr;
    using PCLPointCloud2ConstPtr = PCLPointCloud2::ConstPtr;

    namespace detail
    {

        template <typename PointT> struct FieldCollector
        {
            std::vector<PCLPointField>& fields;

            template <typename Tag> void operator()()
            {
                PCLPointField field;
                field.name = traits::name<PointT, Tag>::value;
                field.offset = static_cast<uindex_t>(traits::offset<PointT, Tag>::value);
                field.datatype = traits::datatype<PointT, Tag>::value;
                field.count = traits::datatype<PointT, Tag>::size;
                fields.push_back(std::move(field));
            }
        };

        template <typename PointT> struct BlobPointReader
        {
            const PCLPointCloud2& blob;
            std::size_t point_offset;
            PointT& point;

            template <typename Tag> void operator()()
            {
                const char* expected_name = traits::name<PointT, Tag>::value;
                const auto field = std::find_if(blob.fields.begin(),
                                                blob.fields.end(),
                                                [expected_name](const auto& candidate)
                                                { return equivalentFieldName(candidate.name, expected_name); });
                if (field == blob.fields.end() || field->count < traits::datatype<PointT, Tag>::size)
                {
                    return;
                }
                using RegisteredType = typename traits::datatype<PointT, Tag>::type;
                using Element = typename traits::decomposeArray<RegisteredType>::type;
                constexpr std::size_t count = traits::datatype<PointT, Tag>::size;
                const std::size_t source_element_size = getFieldSize(field->datatype);
                const std::size_t source_bytes = source_element_size * count;
                if (source_element_size == 0 || point_offset + field->offset + source_bytes > blob.data.size())
                {
                    throw std::invalid_argument("fromPCLPointCloud2: field exceeds point data");
                }
                auto* destination = reinterpret_cast<Element*>(reinterpret_cast<std::uint8_t*>(&point) +
                                                               traits::offset<PointT, Tag>::value);
                const auto* source = blob.data.data() + point_offset + field->offset;
                const bool packed_color = std::string(expected_name) == "rgb" || std::string(expected_name) == "rgba";
                if (packed_color && source_element_size == sizeof(RegisteredType) && count == 1)
                {
                    std::memcpy(destination, source, sizeof(RegisteredType));
                    return;
                }
                if (field->datatype == traits::datatype<PointT, Tag>::value)
                {
                    std::memcpy(destination, source, sizeof(RegisteredType));
                    return;
                }
                for (std::size_t index = 0; index < count; ++index)
                {
                    destination[index] =
                        static_cast<Element>(readNumericField(source + index * source_element_size, field->datatype));
                }
            }
        };

    } // namespace detail

    template <typename PointT> std::vector<PCLPointField> getFields()
    {
        std::vector<PCLPointField> fields;
        detail::FieldCollector<PointT> collector{fields};
        for_each_type<typename traits::fieldList<PointT>::type>(collector);
        return fields;
    }

    template <typename PointT> void toPCLPointCloud2(const PointCloud<PointT>& cloud, PCLPointCloud2& blob)
    {
        if (cloud.size() > std::numeric_limits<uindex_t>::max())
        {
            throw std::overflow_error("toPCLPointCloud2: point count exceeds 32-bit width");
        }
        blob.header = cloud.header;
        blob.width = cloud.width;
        blob.height = cloud.height;
        blob.fields = getFields<PointT>();
        blob.is_bigendian = static_cast<std::uint8_t>(detail::hostIsBigEndian());
        blob.point_step = static_cast<uindex_t>(sizeof(PointT));
        blob.row_step = blob.point_step * blob.width;
        blob.is_dense = static_cast<std::uint8_t>(cloud.is_dense);
        blob.data.resize(cloud.size() * sizeof(PointT));
        for (std::size_t index = 0; index < cloud.size(); ++index)
        {
            std::memcpy(blob.data.data() + index * sizeof(PointT), &cloud.points[index], sizeof(PointT));
        }
    }

    template <typename PointT> void fromPCLPointCloud2(const PCLPointCloud2& blob, PointCloud<PointT>& cloud)
    {
        if (blob.is_bigendian != static_cast<std::uint8_t>(detail::hostIsBigEndian()))
        {
            throw std::invalid_argument("fromPCLPointCloud2: byte swapping is not supported");
        }
        const std::uint64_t point_count = static_cast<std::uint64_t>(blob.width) * blob.height;
        if (point_count > std::numeric_limits<std::size_t>::max() || point_count * blob.point_step > blob.data.size())
        {
            throw std::invalid_argument("fromPCLPointCloud2: inconsistent cloud dimensions");
        }
        PointCloud<PointT> result(static_cast<std::size_t>(point_count));
        result.header = blob.header;
        result.width = blob.width;
        result.height = blob.height;
        result.is_dense = blob.is_dense != 0;
        for (std::size_t index = 0; index < result.size(); ++index)
        {
            detail::BlobPointReader<PointT> reader{blob, index * blob.point_step, result.points[index]};
            for_each_type<typename traits::fieldList<PointT>::type>(reader);
        }
        cloud = std::move(result);
    }

} // namespace plapoint
