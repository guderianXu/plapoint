#include "quality/mesh_quality_utils.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

template <typename Scalar>
void writeMeshPly(const std::filesystem::path& path,
                  const plapoint::test::mesh_quality::Mesh<Scalar>& mesh)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Cannot write PLY file: " + path.string());
    }

    out << std::setprecision(9);
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << mesh.vertices.rows() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "element face " << mesh.faces.rows() << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    for (plamatrix::Index i = 0; i < mesh.vertices.rows(); ++i)
    {
        out << mesh.vertices.getValue(i, 0) << ' '
            << mesh.vertices.getValue(i, 1) << ' '
            << mesh.vertices.getValue(i, 2) << '\n';
    }
    for (plamatrix::Index i = 0; i < mesh.faces.rows(); ++i)
    {
        out << "3 "
            << static_cast<int>(std::round(mesh.faces.getValue(i, 0))) << ' '
            << static_cast<int>(std::round(mesh.faces.getValue(i, 1))) << ' '
            << static_cast<int>(std::round(mesh.faces.getValue(i, 2))) << '\n';
    }
}

void writeMetricJson(std::ostream& out,
                     const char* name,
                     const plapoint::test::mesh_quality::SphereMeshMetrics& metrics,
                     bool trailing_comma)
{
    out << "  \"" << name << "\": {\n"
        << "    \"vertex_count\": " << metrics.vertex_count << ",\n"
        << "    \"face_count\": " << metrics.face_count << ",\n"
        << "    \"invalid_face_count\": " << metrics.invalid_face_count << ",\n"
        << "    \"degenerate_face_count\": " << metrics.degenerate_face_count << ",\n"
        << "    \"degenerate_face_ratio\": " << metrics.degenerate_face_ratio << ",\n"
        << "    \"max_radius_error\": " << metrics.max_radius_error << ",\n"
        << "    \"mean_radius_error\": " << metrics.mean_radius_error << ",\n"
        << "    \"max_abs_coordinate\": " << metrics.max_abs_coordinate << ",\n"
        << "    \"dominant_orientation_ratio\": " << metrics.dominant_orientation_ratio << "\n"
        << "  }";
    if (trailing_comma)
    {
        out << ',';
    }
    out << '\n';
}

std::filesystem::path parseOutputDir(int argc, char** argv)
{
    std::filesystem::path output_dir = "mesh_quality_report";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--output-dir")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("--output-dir requires a path");
            }
            output_dir = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: plapoint_mesh_quality_report [--output-dir DIR]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return output_dir;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        using Scalar = float;
        const auto output_dir = parseOutputDir(argc, argv);
        std::filesystem::create_directories(output_dir);

        const auto marching_cubes = plapoint::test::mesh_quality::generateMarchingCubesSphere<Scalar>(
            Scalar(2), 40);
        const auto marching_cubes_metrics = plapoint::test::mesh_quality::measureSphereMesh(
            marching_cubes.vertices, marching_cubes.faces, Scalar(2));

        const auto poisson = plapoint::test::mesh_quality::generatePoissonSphere<Scalar>(
            Scalar(2), 12, 24, 5, 30);
        const auto poisson_metrics = plapoint::test::mesh_quality::measureSphereMesh(
            poisson.vertices, poisson.faces, Scalar(2));

        writeMeshPly(output_dir / "marching_cubes_sphere.ply", marching_cubes);
        writeMeshPly(output_dir / "poisson_sphere.ply", poisson);

        std::ofstream report(output_dir / "mesh_quality_report.json");
        if (!report)
        {
            throw std::runtime_error("Cannot write report JSON");
        }
        report << std::setprecision(9);
        report << "{\n";
        writeMetricJson(report, "marching_cubes_sphere", marching_cubes_metrics, true);
        writeMetricJson(report, "poisson_sphere", poisson_metrics, false);
        report << "}\n";

        std::cout << "Wrote mesh quality report to " << output_dir.string() << '\n';
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "mesh quality report failed: " << e.what() << '\n';
        return 1;
    }
}
