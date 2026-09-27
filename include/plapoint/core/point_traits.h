#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <boost/mpl/contains.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

namespace plapoint
{

    namespace detail
    {

        template <typename Iterator, typename End> struct ForEachPointType
        {
            template <typename Functor> static void apply(Functor& functor)
            {
                using Tag = typename boost::mpl::deref<Iterator>::type;
                functor.template operator()<Tag>();
                ForEachPointType<typename boost::mpl::next<Iterator>::type, End>::apply(functor);
            }
        };

        template <typename End> struct ForEachPointType<End, End>
        {
            template <typename Functor> static void apply(Functor&)
            {
            }
        };

    } // namespace detail

    template <typename Sequence, typename Functor> void for_each_type(Functor functor)
    {
        using Begin = typename boost::mpl::begin<Sequence>::type;
        using End = typename boost::mpl::end<Sequence>::type;
        detail::ForEachPointType<Begin, End>::apply(functor);
    }

    namespace fields
    {
        struct x;
        struct y;
        struct z;
        struct intensity;
        struct rgb;
        struct rgba;
        struct normal_x;
        struct normal_y;
        struct normal_z;
        struct curvature;
    } // namespace fields

    namespace traits
    {

        template <typename T> struct asEnum;
        template <> struct asEnum<bool>
        {
            static constexpr std::uint8_t value = 11;
        };
        template <> struct asEnum<std::int8_t>
        {
            static constexpr std::uint8_t value = 1;
        };
        template <> struct asEnum<std::uint8_t>
        {
            static constexpr std::uint8_t value = 2;
        };
        template <> struct asEnum<std::int16_t>
        {
            static constexpr std::uint8_t value = 3;
        };
        template <> struct asEnum<std::uint16_t>
        {
            static constexpr std::uint8_t value = 4;
        };
        template <> struct asEnum<std::int32_t>
        {
            static constexpr std::uint8_t value = 5;
        };
        template <> struct asEnum<std::uint32_t>
        {
            static constexpr std::uint8_t value = 6;
        };
        template <> struct asEnum<std::int64_t>
        {
            static constexpr std::uint8_t value = 9;
        };
        template <> struct asEnum<std::uint64_t>
        {
            static constexpr std::uint8_t value = 10;
        };
        template <> struct asEnum<float>
        {
            static constexpr std::uint8_t value = 7;
        };
        template <> struct asEnum<double>
        {
            static constexpr std::uint8_t value = 8;
        };

        template <typename T> inline constexpr std::uint8_t asEnum_v = asEnum<T>::value;

        template <int Value> struct asType;
        template <> struct asType<1>
        {
            using type = std::int8_t;
        };
        template <> struct asType<2>
        {
            using type = std::uint8_t;
        };
        template <> struct asType<3>
        {
            using type = std::int16_t;
        };
        template <> struct asType<4>
        {
            using type = std::uint16_t;
        };
        template <> struct asType<5>
        {
            using type = std::int32_t;
        };
        template <> struct asType<6>
        {
            using type = std::uint32_t;
        };
        template <> struct asType<7>
        {
            using type = float;
        };
        template <> struct asType<8>
        {
            using type = double;
        };
        template <> struct asType<9>
        {
            using type = std::int64_t;
        };
        template <> struct asType<10>
        {
            using type = std::uint64_t;
        };
        template <> struct asType<11>
        {
            using type = bool;
        };
        template <int Value> using asType_t = typename asType<Value>::type;

        template <typename T> struct decomposeArray
        {
            using type = std::remove_all_extents_t<T>;
            static constexpr std::uint32_t value = sizeof(T) / sizeof(type);
        };

        template <typename PointT> struct POD
        {
            using type = PointT;
        };

        template <typename PointT, typename Tag, int Dummy = 0>
        struct name : name<typename POD<PointT>::type, Tag, Dummy>
        {
        };

        template <typename PointT, typename Tag> struct offset : offset<typename POD<PointT>::type, Tag>
        {
        };

        template <typename PointT, typename Tag> struct datatype : datatype<typename POD<PointT>::type, Tag>
        {
        };

        template <typename PointT> struct fieldList : fieldList<typename POD<PointT>::type>
        {
        };

        template <typename PointT, typename Tag>
        struct has_field : boost::mpl::contains<typename fieldList<PointT>::type, Tag>::type
        {
        };

        template <typename PointT, typename Tag> inline constexpr bool has_field_v = has_field<PointT, Tag>::value;

        template <typename PointT>
        struct has_xyz : std::bool_constant<has_field_v<PointT, fields::x> && has_field_v<PointT, fields::y> &&
                                            has_field_v<PointT, fields::z>>
        {
        };

    } // namespace traits
} // namespace plapoint

#define PLAPOINT_REGISTER_POINT_STRUCT_X(type, member, tag) ((type, member, tag)) PLAPOINT_REGISTER_POINT_STRUCT_Y
#define PLAPOINT_REGISTER_POINT_STRUCT_Y(type, member, tag) ((type, member, tag)) PLAPOINT_REGISTER_POINT_STRUCT_X
#define PLAPOINT_REGISTER_POINT_STRUCT_X0
#define PLAPOINT_REGISTER_POINT_STRUCT_Y0

#define PLAPOINT_REGISTER_FIELD_TAG(r, point, elem) struct BOOST_PP_TUPLE_ELEM(3, 2, elem);

#define PLAPOINT_REGISTER_FIELD_NAME(r, point, elem)                                                                   \
    template <int Dummy> struct name<point, ::plapoint::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem), Dummy>                \
    {                                                                                                                  \
        inline static constexpr char value[] = BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 2, elem));                      \
    };

#define PLAPOINT_REGISTER_FIELD_OFFSET(r, point, elem)                                                                 \
    template <> struct offset<point, ::plapoint::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)>                              \
    {                                                                                                                  \
        static constexpr std::size_t value = offsetof(point, BOOST_PP_TUPLE_ELEM(3, 1, elem));                         \
    };

#define PLAPOINT_REGISTER_FIELD_DATATYPE(r, point, elem)                                                               \
    template <> struct datatype<point, ::plapoint::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)>                            \
    {                                                                                                                  \
        using type = typename boost::mpl::identity<BOOST_PP_TUPLE_ELEM(3, 0, elem)>::type;                             \
        using decomposed = decomposeArray<type>;                                                                       \
        static constexpr std::uint8_t value = asEnum<typename decomposed::type>::value;                                \
        static constexpr std::uint32_t size = decomposed::value;                                                       \
    };

#define PLAPOINT_REGISTER_TAG_OP(s, data, elem) ::plapoint::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)
#define PLAPOINT_EXTRACT_TAGS(seq) BOOST_PP_SEQ_TRANSFORM(PLAPOINT_REGISTER_TAG_OP, _, seq)

#define PLAPOINT_REGISTER_POINT_FIELD_LIST(point, seq)                                                                 \
    template <> struct fieldList<point>                                                                                \
    {                                                                                                                  \
        using type = boost::mpl::vector<BOOST_PP_SEQ_ENUM(seq)>;                                                       \
    };

#define PLAPOINT_REGISTER_POINT_STRUCT_IMPL(point, seq)                                                                \
    namespace plapoint                                                                                                 \
    {                                                                                                                  \
        namespace fields                                                                                               \
        {                                                                                                              \
            BOOST_PP_SEQ_FOR_EACH(PLAPOINT_REGISTER_FIELD_TAG, point, seq)                                             \
        }                                                                                                              \
        namespace traits                                                                                               \
        {                                                                                                              \
            BOOST_PP_SEQ_FOR_EACH(PLAPOINT_REGISTER_FIELD_NAME, point, seq)                                            \
            BOOST_PP_SEQ_FOR_EACH(PLAPOINT_REGISTER_FIELD_OFFSET, point, seq)                                          \
            BOOST_PP_SEQ_FOR_EACH(PLAPOINT_REGISTER_FIELD_DATATYPE, point, seq)                                        \
            PLAPOINT_REGISTER_POINT_FIELD_LIST(point, PLAPOINT_EXTRACT_TAGS(seq))                                      \
        }                                                                                                              \
    }

// Kept source-compatible with the point registration macros used by existing projects.
#define POINT_CLOUD_REGISTER_POINT_STRUCT(point, fields)                                                               \
    PLAPOINT_REGISTER_POINT_STRUCT_IMPL(point, BOOST_PP_CAT(PLAPOINT_REGISTER_POINT_STRUCT_X fields, 0))

#define POINT_CLOUD_REGISTER_POINT_WRAPPER(wrapper, pod)                                                               \
    static_assert(sizeof(wrapper) == sizeof(pod), "point wrapper and storage type sizes differ");                      \
    namespace plapoint                                                                                                 \
    {                                                                                                                  \
        namespace traits                                                                                               \
        {                                                                                                              \
            template <> struct POD<wrapper>                                                                            \
            {                                                                                                          \
                using type = pod;                                                                                      \
            };                                                                                                         \
        }                                                                                                              \
    }
