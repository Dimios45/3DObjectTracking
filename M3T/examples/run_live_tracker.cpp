// Live tracker: D=detect, T=track+record, Q=quit+save MP4
#include <filesystem/filesystem.h>
#include <m3t/generator.h>
#include <m3t/tracker.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: run_live_tracker <configfile> [output_dir]\n";
    return -1;
  }
  const std::filesystem::path configfile_path{argv[1]};

  // Auto-generate timestamped output dir if not provided
  std::filesystem::path output_dir;
  if (argc >= 3) {
    output_dir = argv[2];
  } else {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "recording_%Y%m%d_%H%M%S",
                  std::localtime(&t));
    output_dir = configfile_path.parent_path() / buf;
  }

  std::shared_ptr<m3t::Tracker> tracker_ptr;
  if (!GenerateConfiguredTracker(configfile_path, &tracker_ptr)) return -1;
  tracker_ptr->set_recording_directory(output_dir);
  if (!tracker_ptr->SetUp()) return -1;

  std::cout << "\nControls:\n"
            << "  D  - Detect (initialize pose)\n"
            << "  T  - Start tracking + recording\n"
            << "  S  - Stop tracking\n"
            << "  Q  - Quit and save MP4\n\n";

  tracker_ptr->RunTrackerProcess(false, false);

  // Count saved PNG frames
  int n_frames = 0;
  std::string prefix;
  if (std::filesystem::exists(output_dir)) {
    for (auto &p : std::filesystem::directory_iterator(output_dir)) {
      if (p.path().extension() == ".png") {
        n_frames++;
        if (prefix.empty()) {
          std::string fname = p.path().filename().string();
          auto pos = fname.find("_image_");
          if (pos != std::string::npos)
            prefix = fname.substr(0, pos + 7);
        }
      }
    }
  }

  if (n_frames == 0) {
    std::cout << "No frames recorded.\n";
    return 0;
  }

  std::cout << "Saved " << n_frames << " frames. Creating MP4...\n";

  // Find start frame number
  int start_num = INT_MAX;
  for (auto &p : std::filesystem::directory_iterator(output_dir)) {
    if (p.path().extension() == ".png") {
      std::string fname = p.path().stem().string();
      auto pos = fname.rfind('_');
      if (pos != std::string::npos) {
        try {
          int n = std::stoi(fname.substr(pos + 1));
          start_num = std::min(start_num, n);
        } catch (...) {}
      }
    }
  }

  std::string mp4 = output_dir.string() + "/tracking.mp4";
  std::string cmd = "ffmpeg -y -framerate 30 -start_number " +
                    std::to_string(start_num) + " -i " +
                    output_dir.string() + "/" + prefix + "%d.png" +
                    " -vcodec libx264 -pix_fmt yuv420p " + mp4 +
                    " && echo 'Saved: " + mp4 + "'";
  std::system(cmd.c_str());

  return 0;
}
