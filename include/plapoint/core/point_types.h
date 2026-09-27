#pragma once

#include <cstdint>

#include <Eigen/Core>

#include <plapoint/core/point_traits.h>

namespace plapoint
{

    using Vector2fMap = Eigen::Map<Eigen::Vector2f>;
    using Vector2fMapConst = const Eigen::Map<const Eigen::Vector2f>;
    using Vector3fMap = Eigen::Map<Eigen::Vector3f>;
    using Vector3fMapConst = const Eigen::Map<const Eigen::Vector3f>;
    using Vector4fMap = Eigen::Map<Eigen::Vector4f, Eigen::Aligned>;
    using Vector4fMapConst = const Eigen::Map<const Eigen::Vector4f, Eigen::Aligned>;
    using Array3fMap = Eigen::Map<Eigen::Array3f>;
    using Array3fMapConst = const Eigen::Map<const Eigen::Array3f>;
    using Array4fMap = Eigen::Map<Eigen::Array4f, Eigen::Aligned>;
    using Array4fMapConst = const Eigen::Map<const Eigen::Array4f, Eigen::Aligned>;
    using Vector3cMap = Eigen::Map<Eigen::Vector<std::uint8_t, 3>>;
    using Vector3cMapConst = const Eigen::Map<const Eigen::Vector<std::uint8_t, 3>>;
    using Vector4cMap = Eigen::Map<Eigen::Vector<std::uint8_t, 4>>;
    using Vector4cMapConst = const Eigen::Map<const Eigen::Vector<std::uint8_t, 4>>;

    namespace detail
    {

        struct alignas(16) Point4f
        {
            union
            {
                float data[4];
                struct
                {
                    float x;
                    float y;
                    float z;
                };
            };

            Vector2fMap getVector2fMap()
            {
                return Vector2fMap(data);
            }
            Vector2fMapConst getVector2fMap() const
            {
                return Vector2fMapConst(data);
            }
            Vector3fMap getVector3fMap()
            {
                return Vector3fMap(data);
            }
            Vector3fMapConst getVector3fMap() const
            {
                return Vector3fMapConst(data);
            }
            Vector4fMap getVector4fMap()
            {
                return Vector4fMap(data);
            }
            Vector4fMapConst getVector4fMap() const
            {
                return Vector4fMapConst(data);
            }
            Array3fMap getArray3fMap()
            {
                return Array3fMap(data);
            }
            Array3fMapConst getArray3fMap() const
            {
                return Array3fMapConst(data);
            }
            Array4fMap getArray4fMap()
            {
                return Array4fMap(data);
            }
            Array4fMapConst getArray4fMap() const
            {
                return Array4fMapConst(data);
            }
        };

        struct alignas(16) Normal4f
        {
            union
            {
                float data_n[4];
                float normal[3];
                struct
                {
                    float normal_x;
                    float normal_y;
                    float normal_z;
                };
            };

            Vector3fMap getNormalVector3fMap()
            {
                return Vector3fMap(data_n);
            }
            Vector3fMapConst getNormalVector3fMap() const
            {
                return Vector3fMapConst(data_n);
            }
            Vector4fMap getNormalVector4fMap()
            {
                return Vector4fMap(data_n);
            }
            Vector4fMapConst getNormalVector4fMap() const
            {
                return Vector4fMapConst(data_n);
            }
        };

        struct Color4f
        {
            union
            {
                struct
                {
                    std::uint8_t b;
                    std::uint8_t g;
                    std::uint8_t r;
                    std::uint8_t a;
                };
                float rgb;
                std::uint32_t rgba;
            };

            Eigen::Vector3i getRGBVector3i()
            {
                return Eigen::Vector3i(r, g, b);
            }
            const Eigen::Vector3i getRGBVector3i() const
            {
                return Eigen::Vector3i(r, g, b);
            }
            Eigen::Vector4i getRGBVector4i()
            {
                return Eigen::Vector4i(r, g, b, a);
            }
            const Eigen::Vector4i getRGBVector4i() const
            {
                return Eigen::Vector4i(r, g, b, a);
            }
            Eigen::Vector4i getRGBAVector4i()
            {
                return getRGBVector4i();
            }
            const Eigen::Vector4i getRGBAVector4i() const
            {
                return getRGBVector4i();
            }
            Vector3cMap getBGRVector3cMap()
            {
                return Vector3cMap(reinterpret_cast<std::uint8_t*>(&rgba));
            }
            Vector3cMapConst getBGRVector3cMap() const
            {
                return Vector3cMapConst(reinterpret_cast<const std::uint8_t*>(&rgba));
            }
            Vector4cMap getBGRAVector4cMap()
            {
                return Vector4cMap(reinterpret_cast<std::uint8_t*>(&rgba));
            }
            Vector4cMapConst getBGRAVector4cMap() const
            {
                return Vector4cMapConst(reinterpret_cast<const std::uint8_t*>(&rgba));
            }
        };

    } // namespace detail

    /// PCL-compatible, 16-byte aligned XYZ point with homogeneous coordinate data[3] = 1.
    struct alignas(16) PointXYZ : detail::Point4f
    {
        PointXYZ() : PointXYZ(0.0f, 0.0f, 0.0f)
        {
        }
        PointXYZ(float x_value, float y_value, float z_value)
        {
            data[0] = x_value;
            data[1] = y_value;
            data[2] = z_value;
            data[3] = 1.0f;
        }
    };

    /// Double-precision XYZ extension for georeferenced planetary coordinates.
    struct PointXYZd
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        PointXYZd() = default;
        PointXYZd(double x_value, double y_value, double z_value) : x(x_value), y(y_value), z(z_value)
        {
        }
    };

    /// PCL-compatible XYZ and packed RGB point.
    struct alignas(16) PointXYZRGB : detail::Point4f, detail::Color4f
    {
        PointXYZRGB() : PointXYZRGB(0.0f, 0.0f, 0.0f)
        {
        }
        PointXYZRGB(std::uint8_t red, std::uint8_t green, std::uint8_t blue)
            : PointXYZRGB(0.0f, 0.0f, 0.0f, red, green, blue)
        {
        }
        PointXYZRGB(float x_value, float y_value, float z_value) : PointXYZRGB(x_value, y_value, z_value, 0, 0, 0)
        {
        }
        PointXYZRGB(
            float x_value, float y_value, float z_value, std::uint8_t red, std::uint8_t green, std::uint8_t blue)
        {
            data[0] = x_value;
            data[1] = y_value;
            data[2] = z_value;
            data[3] = 1.0f;
            b = blue;
            g = green;
            r = red;
            a = 255;
        }
    };

    /// XYZ point with a floating-point intensity value.
    struct alignas(16) PointXYZI : detail::Point4f
    {
        union
        {
            struct
            {
                float intensity;
            };
            float data_c[4];
        };

        explicit PointXYZI(float intensity_value = 0.0f) : PointXYZI(0.0f, 0.0f, 0.0f, intensity_value)
        {
        }
        PointXYZI(float x_value, float y_value, float z_value, float intensity_value = 0.0f)
        {
            data[0] = x_value;
            data[1] = y_value;
            data[2] = z_value;
            data[3] = 1.0f;
            data_c[0] = intensity_value;
            data_c[1] = data_c[2] = data_c[3] = 0.0f;
        }
    };

    /// XYZ point with packed RGBA color.
    struct alignas(16) PointXYZRGBA : detail::Point4f, detail::Color4f
    {
        PointXYZRGBA() : PointXYZRGBA(0, 0, 0, 255)
        {
        }
        PointXYZRGBA(std::uint8_t red, std::uint8_t green, std::uint8_t blue, std::uint8_t alpha)
            : PointXYZRGBA(0.0f, 0.0f, 0.0f, red, green, blue, alpha)
        {
        }
        PointXYZRGBA(float x_value, float y_value, float z_value)
            : PointXYZRGBA(x_value, y_value, z_value, 0, 0, 0, 255)
        {
        }
        PointXYZRGBA(float x_value,
                     float y_value,
                     float z_value,
                     std::uint8_t red,
                     std::uint8_t green,
                     std::uint8_t blue,
                     std::uint8_t alpha)
        {
            data[0] = x_value;
            data[1] = y_value;
            data[2] = z_value;
            data[3] = 1.0f;
            b = blue;
            g = green;
            r = red;
            a = alpha;
        }
    };

    /// PCL-compatible normal, including curvature and four-float normal storage.
    struct alignas(16) Normal : detail::Normal4f
    {
        union
        {
            float data_c[4];
            struct
            {
                float curvature;
            };
        };

        explicit Normal(float curvature_value = 0.0f) : Normal(0.0f, 0.0f, 0.0f, curvature_value)
        {
        }
        Normal(float nx, float ny, float nz, float curvature_value = 0.0f)
        {
            data_n[0] = nx;
            data_n[1] = ny;
            data_n[2] = nz;
            data_n[3] = 0.0f;
            data_c[0] = curvature_value;
            data_c[1] = data_c[2] = data_c[3] = 0.0f;
        }
    };

    /// PCL-compatible XYZ point with normal and curvature.
    struct alignas(16) PointNormal : detail::Point4f, detail::Normal4f
    {
        union
        {
            float data_c[4];
            struct
            {
                float curvature;
            };
        };

        explicit PointNormal(float curvature_value = 0.0f)
            : PointNormal(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, curvature_value)
        {
        }
        PointNormal(float x_value, float y_value, float z_value)
            : PointNormal(x_value, y_value, z_value, 0.0f, 0.0f, 0.0f)
        {
        }
        PointNormal(
            float x_value, float y_value, float z_value, float nx, float ny, float nz, float curvature_value = 0.0f)
        {
            data[0] = x_value;
            data[1] = y_value;
            data[2] = z_value;
            data[3] = 1.0f;
            data_n[0] = nx;
            data_n[1] = ny;
            data_n[2] = nz;
            data_n[3] = 0.0f;
            data_c[0] = curvature_value;
            data_c[1] = data_c[2] = data_c[3] = 0.0f;
        }
    };

    static_assert(sizeof(PointXYZ) == 16 && alignof(PointXYZ) == 16);
    static_assert(sizeof(PointXYZI) == 32 && alignof(PointXYZI) == 16);
    static_assert(sizeof(PointXYZRGB) == 32 && alignof(PointXYZRGB) == 16);
    static_assert(sizeof(PointXYZRGBA) == 32 && alignof(PointXYZRGBA) == 16);
    static_assert(sizeof(Normal) == 32 && alignof(Normal) == 16);
    static_assert(sizeof(PointNormal) == 48 && alignof(PointNormal) == 16);

} // namespace plapoint

#ifndef PCL_MAKE_ALIGNED_OPERATOR_NEW
#define PCL_MAKE_ALIGNED_OPERATOR_NEW EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif

#ifndef PCL_ADD_POINT4D
#define PCL_ADD_POINT4D                                                                                                \
    union EIGEN_ALIGN16                                                                                                \
    {                                                                                                                  \
        float data[4];                                                                                                 \
        struct                                                                                                         \
        {                                                                                                              \
            float x;                                                                                                   \
            float y;                                                                                                   \
            float z;                                                                                                   \
        };                                                                                                             \
    };                                                                                                                 \
    ::plapoint::Vector2fMap getVector2fMap()                                                                           \
    {                                                                                                                  \
        return ::plapoint::Vector2fMap(data);                                                                          \
    }                                                                                                                  \
    ::plapoint::Vector2fMapConst getVector2fMap() const                                                                \
    {                                                                                                                  \
        return ::plapoint::Vector2fMapConst(data);                                                                     \
    }                                                                                                                  \
    ::plapoint::Vector3fMap getVector3fMap()                                                                           \
    {                                                                                                                  \
        return ::plapoint::Vector3fMap(data);                                                                          \
    }                                                                                                                  \
    ::plapoint::Vector3fMapConst getVector3fMap() const                                                                \
    {                                                                                                                  \
        return ::plapoint::Vector3fMapConst(data);                                                                     \
    }                                                                                                                  \
    ::plapoint::Vector4fMap getVector4fMap()                                                                           \
    {                                                                                                                  \
        return ::plapoint::Vector4fMap(data);                                                                          \
    }                                                                                                                  \
    ::plapoint::Vector4fMapConst getVector4fMap() const                                                                \
    {                                                                                                                  \
        return ::plapoint::Vector4fMapConst(data);                                                                     \
    }                                                                                                                  \
    ::plapoint::Array3fMap getArray3fMap()                                                                             \
    {                                                                                                                  \
        return ::plapoint::Array3fMap(data);                                                                           \
    }                                                                                                                  \
    ::plapoint::Array3fMapConst getArray3fMap() const                                                                  \
    {                                                                                                                  \
        return ::plapoint::Array3fMapConst(data);                                                                      \
    }                                                                                                                  \
    ::plapoint::Array4fMap getArray4fMap()                                                                             \
    {                                                                                                                  \
        return ::plapoint::Array4fMap(data);                                                                           \
    }                                                                                                                  \
    ::plapoint::Array4fMapConst getArray4fMap() const                                                                  \
    {                                                                                                                  \
        return ::plapoint::Array4fMapConst(data);                                                                      \
    }
#endif

#ifndef PCL_ADD_NORMAL4D
#define PCL_ADD_NORMAL4D                                                                                               \
    union EIGEN_ALIGN16                                                                                                \
    {                                                                                                                  \
        float data_n[4];                                                                                               \
        float normal[3];                                                                                               \
        struct                                                                                                         \
        {                                                                                                              \
            float normal_x;                                                                                            \
            float normal_y;                                                                                            \
            float normal_z;                                                                                            \
        };                                                                                                             \
    };                                                                                                                 \
    ::plapoint::Vector3fMap getNormalVector3fMap()                                                                     \
    {                                                                                                                  \
        return ::plapoint::Vector3fMap(data_n);                                                                        \
    }                                                                                                                  \
    ::plapoint::Vector3fMapConst getNormalVector3fMap() const                                                          \
    {                                                                                                                  \
        return ::plapoint::Vector3fMapConst(data_n);                                                                   \
    }                                                                                                                  \
    ::plapoint::Vector4fMap getNormalVector4fMap()                                                                     \
    {                                                                                                                  \
        return ::plapoint::Vector4fMap(data_n);                                                                        \
    }                                                                                                                  \
    ::plapoint::Vector4fMapConst getNormalVector4fMap() const                                                          \
    {                                                                                                                  \
        return ::plapoint::Vector4fMapConst(data_n);                                                                   \
    }
#endif

#ifndef PCL_ADD_RGB
#define PCL_ADD_RGB                                                                                                    \
    union                                                                                                              \
    {                                                                                                                  \
        union                                                                                                          \
        {                                                                                                              \
            struct                                                                                                     \
            {                                                                                                          \
                std::uint8_t b;                                                                                        \
                std::uint8_t g;                                                                                        \
                std::uint8_t r;                                                                                        \
                std::uint8_t a;                                                                                        \
            };                                                                                                         \
            float rgb;                                                                                                 \
        };                                                                                                             \
        std::uint32_t rgba;                                                                                            \
    };                                                                                                                 \
    Eigen::Vector3i getRGBVector3i() const                                                                             \
    {                                                                                                                  \
        return Eigen::Vector3i(r, g, b);                                                                               \
    }                                                                                                                  \
    Eigen::Vector4i getRGBVector4i() const                                                                             \
    {                                                                                                                  \
        return Eigen::Vector4i(r, g, b, a);                                                                            \
    }                                                                                                                  \
    Eigen::Vector4i getRGBAVector4i() const                                                                            \
    {                                                                                                                  \
        return Eigen::Vector4i(r, g, b, a);                                                                            \
    }                                                                                                                  \
    ::plapoint::Vector3cMap getBGRVector3cMap()                                                                        \
    {                                                                                                                  \
        return ::plapoint::Vector3cMap(reinterpret_cast<std::uint8_t*>(&rgba));                                        \
    }                                                                                                                  \
    ::plapoint::Vector3cMapConst getBGRVector3cMap() const                                                             \
    {                                                                                                                  \
        return ::plapoint::Vector3cMapConst(reinterpret_cast<const std::uint8_t*>(&rgba));                             \
    }                                                                                                                  \
    ::plapoint::Vector4cMap getBGRAVector4cMap()                                                                       \
    {                                                                                                                  \
        return ::plapoint::Vector4cMap(reinterpret_cast<std::uint8_t*>(&rgba));                                        \
    }                                                                                                                  \
    ::plapoint::Vector4cMapConst getBGRAVector4cMap() const                                                            \
    {                                                                                                                  \
        return ::plapoint::Vector4cMapConst(reinterpret_cast<const std::uint8_t*>(&rgba));                             \
    }
#endif

#ifndef PCL_ADD_INTENSITY
#define PCL_ADD_INTENSITY                                                                                              \
    struct                                                                                                             \
    {                                                                                                                  \
        float intensity;                                                                                               \
    };
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointXYZ, (float, x, x)(float, y, y)(float, z, z))

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointXYZd, (double, x, x)(double, y, y)(double, z, z))

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointXYZI,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity))

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointXYZRGB, (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb))

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointXYZRGBA,
                                  (float, x, x)(float, y, y)(float, z, z)(std::uint32_t, rgba, rgba))

POINT_CLOUD_REGISTER_POINT_STRUCT(
    plapoint::Normal,
    (float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z)(float, curvature, curvature))

POINT_CLOUD_REGISTER_POINT_STRUCT(plapoint::PointNormal,
                                  (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
                                      float, normal_y, normal_y)(float, normal_z, normal_z)(float,
                                                                                            curvature,
                                                                                            curvature))

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
