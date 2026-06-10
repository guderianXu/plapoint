# Real Reconstruction Reference Data

These files are reference outputs generated from the PlaScan sample imagery in:

- `/home/xjw/code/mygithub/plascan/testData/img`
- `/home/xjw/code/mygithub/plascan/testData/tsai`

The source images were scaled during reconstruction, and the generated non-empty
PLY outputs were copied here as future regression and comparison data for
PlaPoint I/O and point-cloud attribute tests.

## Contents

- `merged/merged_dense_filtered.ply`: merged XYZ/error point cloud, 252224 points.
- `merged/merged_dense_gray.ply`: merged XYZ/error/intensity point cloud, 252224 points.
- `pairs/002_001/cloud.ply`: pair point cloud, 61161 points.
- `pairs/002_001/cloud_gray.ply`: pair point cloud with grayscale intensity, 61161 points.
- `pairs/003_002/cloud.ply`: pair point cloud, 64011 points.
- `pairs/003_002/cloud_gray.ply`: pair point cloud with grayscale intensity, 64011 points.
- `pairs/004_003/cloud.ply`: pair point cloud, 64999 points.
- `pairs/004_003/cloud_gray.ply`: pair point cloud with grayscale intensity, 64999 points.
- `pairs/005_004/cloud.ply`: pair point cloud, 62053 points.
- `pairs/005_004/cloud_gray.ply`: pair point cloud with grayscale intensity, 62053 points.
- `summary.json`, `merge_quality.json`, `gray_summary.json`: generation summaries and basic validation metadata.
- `image_camera_list.txt`: image/camera pairing used for generation.

The `001_002/cloud.ply` output was intentionally not copied because it contains
zero vertices.
