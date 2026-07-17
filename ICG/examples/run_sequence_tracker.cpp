// Autostart tracker for recorded sequences - detects and starts tracking on frame 0
#include <filesystem/filesystem.h>
#include <icg/generator.h>
#include <icg/tracker.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Not enough arguments: Provide configfile_path";
    return -1;
  }
  const std::filesystem::path configfile_path{argv[1]};

  std::shared_ptr<icg::Tracker> tracker_ptr;
  if (!GenerateConfiguredTracker(configfile_path, &tracker_ptr)) return -1;

  if (!tracker_ptr->SetUp()) return -1;
  if (!tracker_ptr->RunTrackerProcess(true, true)) return -1;
  return 0;
}
