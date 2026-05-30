// Ball tracking on video sequence using StaticDetector — ICG
#include <icg/body.h>
#include <icg/common.h>
#include <icg/loader_camera.h>
#include <icg/normal_viewer.h>
#include <icg/optimizer.h>
#include <icg/region_modality.h>
#include <icg/region_model.h>
#include <icg/renderer_geometry.h>
#include <icg/static_detector.h>
#include <icg/tracker.h>
#include <memory>

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: run_ball_tracking "
                 "<camera.yaml> <body.yaml> <detector.yaml> <temp_dir> <output_frames_dir>\n";
    return -1;
  }
  const std::filesystem::path camera_metafile{argv[1]};
  const std::filesystem::path body_metafile{argv[2]};
  const std::filesystem::path detector_metafile{argv[3]};
  const std::filesystem::path temp_directory{argv[4]};
  const std::filesystem::path output_frames_dir{argv[5]};
  std::filesystem::create_directories(output_frames_dir);

  auto tracker_ptr{std::make_shared<icg::Tracker>("tracker")};
  auto renderer_geometry_ptr{
      std::make_shared<icg::RendererGeometry>("renderer_geometry")};

  auto camera_ptr{std::make_shared<icg::LoaderColorCamera>(
      "color_camera", camera_metafile)};

  auto viewer_ptr{std::make_shared<icg::NormalColorViewer>(
      "viewer", camera_ptr, renderer_geometry_ptr)};
  viewer_ptr->StartSavingImages(output_frames_dir, "png");
  tracker_ptr->AddViewer(viewer_ptr);

  auto body_ptr{std::make_shared<icg::Body>("ball", body_metafile)};
  renderer_geometry_ptr->AddBody(body_ptr);

  auto region_model_ptr{std::make_shared<icg::RegionModel>(
      "region_model", body_ptr, temp_directory / "region_model.bin")};

  auto region_modality_ptr{std::make_shared<icg::RegionModality>(
      "region_modality", body_ptr, camera_ptr, region_model_ptr)};

  // ICG: Optimizer takes only name, body linked via modalities
  auto optimizer_ptr{std::make_shared<icg::Optimizer>("optimizer")};
  optimizer_ptr->AddModality(region_modality_ptr);
  tracker_ptr->AddOptimizer(optimizer_ptr);

  auto detector_ptr{std::make_shared<icg::StaticDetector>(
      "detector", detector_metafile, body_ptr)};
  tracker_ptr->AddDetector(detector_ptr);

  if (!tracker_ptr->SetUp()) return -1;
  if (!tracker_ptr->RunTrackerProcess(true, true)) return -1;
  return 0;
}
