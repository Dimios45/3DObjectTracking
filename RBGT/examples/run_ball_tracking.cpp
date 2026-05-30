// Ball tracking on video sequence — RBGT (2020)
#include <rbgt/body.h>
#include <rbgt/common.h>
#include <rbgt/image_loader_camera.h>
#include <rbgt/model.h>
#include <rbgt/normal_image_viewer.h>
#include <rbgt/region_modality.h>
#include <rbgt/renderer_geometry.h>
#include <rbgt/tracker.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: run_ball_tracking "
                 "<rbgt_frames_dir> <model_dir> <output_frames_dir>\n";
    return -1;
  }
  const std::filesystem::path frames_dir{argv[1]};
  const std::filesystem::path model_dir{argv[2]};
  const std::filesystem::path output_frames_dir{argv[3]};
  std::filesystem::create_directories(output_frames_dir);
  std::filesystem::create_directories(model_dir);

  // Tracker — no name argument in RBGT
  auto tracker_ptr{std::make_shared<rbgt::Tracker>()};
  auto renderer_geometry_ptr{std::make_shared<rbgt::RendererGeometry>()};

  // Camera — Init reads metadata file + loads first image
  // Filenames: color_camera_image_1.png .. color_camera_image_81.png
  auto camera_ptr{std::make_shared<rbgt::ImageLoaderCamera>()};
  camera_ptr->set_load_image_type("png");
  if (!camera_ptr->Init("color_camera", frames_dir, "color_camera", 1, 81)) {
    std::cerr << "Camera init failed\n"; return -1;
  }

  // Viewer
  auto viewer_ptr{std::make_shared<rbgt::NormalImageViewer>()};
  if (!viewer_ptr->Init("viewer", renderer_geometry_ptr, camera_ptr)) {
    std::cerr << "Viewer init failed\n"; return -1;
  }
  viewer_ptr->StartSavingImages(output_frames_dir);
  tracker_ptr->AddViewer(viewer_ptr);

  // Body — same convention as SRT3D: world2body_pose
  const std::filesystem::path sphere_obj{
      "/mnt/data/mritunjoyh/Wan2.1/tracking/ball/sphere.obj"};
  rbgt::Transform3fA geometry2body{rbgt::Transform3fA::Identity()};
  auto body_ptr{std::make_shared<rbgt::Body>(
      "ball", sphere_obj, 1.0f, true, true, 0.1f, geometry2body)};

  rbgt::Transform3fA world2body{rbgt::Transform3fA::Identity()};
  world2body.translation() = Eigen::Vector3f{0.013202f, -0.027180f, -0.232974f};
  body_ptr->set_world2body_pose(world2body);
  body_ptr->set_occlusion_mask_id(1);
  renderer_geometry_ptr->AddBody(body_ptr);

  // Model — load if exists, otherwise generate and save
  std::string model_name{"ball_model"};
  auto model_ptr{std::make_shared<rbgt::Model>(model_name)};
  if (!model_ptr->LoadModel(model_dir, model_name)) {
    std::cout << "Generating model..." << std::endl;
    // sphere_radius=0.8, n_divides=4, n_points=200
    model_ptr->GenerateModel(*body_ptr, 0.8f, 4, 200);
    model_ptr->SaveModel(model_dir, model_name);
    std::cout << "Model saved." << std::endl;
  } else {
    std::cout << "Model loaded from cache." << std::endl;
  }

  // Region modality
  auto region_modality_ptr{std::make_shared<rbgt::RegionModality>()};
  if (!region_modality_ptr->Init("region_modality", body_ptr,
                                  model_ptr, camera_ptr)) {
    std::cerr << "RegionModality init failed\n"; return -1;
  }
  tracker_ptr->AddRegionModality(region_modality_ptr);

  // StartTracker(true) = start tracking immediately
  std::cout << "Starting tracker..." << std::endl;
  tracker_ptr->StartTracker(true);
  return 0;
}
