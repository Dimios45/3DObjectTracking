// Ball tracking on video sequence — SRT3D (2022)
#include <srt3d/body.h>
#include <srt3d/common.h>
#include <srt3d/loader_camera.h>
#include <srt3d/model.h>
#include <srt3d/normal_viewer.h>
#include <srt3d/region_modality.h>
#include <srt3d/renderer_geometry.h>
#include <srt3d/tracker.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: run_ball_tracking "
                 "<frames_dir> <model_dir> <output_frames_dir>\n";
    return -1;
  }
  const std::filesystem::path frames_dir{argv[1]};
  const std::filesystem::path model_dir{argv[2]};
  const std::filesystem::path output_frames_dir{argv[3]};
  std::filesystem::create_directories(model_dir);
  std::filesystem::create_directories(output_frames_dir);

  // Tracker
  auto tracker_ptr{std::make_shared<srt3d::Tracker>("tracker")};

  // RendererGeometry — shared between viewer and model rendering
  auto renderer_geometry_ptr{
      std::make_shared<srt3d::RendererGeometry>("renderer_geometry")};

  // Camera — inline intrinsics, zero-padded filenames (0001..0081)
  srt3d::Intrinsics intrinsics{600.0f, 600.0f, 360.0f, 272.0f, 720, 544};
  auto camera_ptr{std::make_shared<srt3d::LoaderCamera>(
      "camera", frames_dir, intrinsics,
      "color_camera_image_", 1, 4, "", "png")};

  // Viewer — display off (headless), save frames
  auto viewer_ptr{std::make_shared<srt3d::NormalViewer>(
      "viewer", camera_ptr, renderer_geometry_ptr)};
  viewer_ptr->set_display_images(false);
  viewer_ptr->StartSavingImages(output_frames_dir, "png");
  tracker_ptr->AddViewer(viewer_ptr);

  // Body — max_diameter must exceed 2*radius = 0.07m
  const std::filesystem::path sphere_obj{
      "/mnt/data/mritunjoyh/Wan2.1/tracking/ball/sphere.obj"};
  srt3d::Transform3fA geometry2body{srt3d::Transform3fA::Identity()};
  auto body_ptr{std::make_shared<srt3d::Body>(
      "ball", sphere_obj, 1.0f, true, true, 0.1f, geometry2body)};

  // world2body: inverse of body2world (tx=-0.013202, ty=0.027180, tz=0.232974)
  srt3d::Transform3fA world2body{srt3d::Transform3fA::Identity()};
  world2body.translation() = Eigen::Vector3f{0.013202f, -0.027180f, -0.232974f};
  body_ptr->set_world2body_pose(world2body);
  renderer_geometry_ptr->AddBody(body_ptr);

  // Model — constructor triggers SetUp: loads from file or generates+saves
  auto model_ptr{std::make_shared<srt3d::Model>(
      "ball_model", body_ptr, model_dir, "ball_model.bin")};

  // Region modality — note arg order: body, model, camera
  auto region_modality_ptr{std::make_shared<srt3d::RegionModality>(
      "region_modality", body_ptr, model_ptr, camera_ptr)};
  tracker_ptr->AddRegionModality(region_modality_ptr);

  // SetUp all objects, then start tracking immediately
  std::cout << "Setting up tracker..." << std::endl;
  if (!tracker_ptr->SetUpTracker()) {
    std::cerr << "SetUpTracker failed\n"; return -1;
  }
  std::cout << "Starting tracker..." << std::endl;
  tracker_ptr->StartTracker(true);
  std::cout << "Done." << std::endl;
  return 0;
}
