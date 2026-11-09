// sdi-classify MODEL IMAGE...: print the predicted class and its probability per image.
// sdi-classify --bench N MODEL IMAGE: time N single-image inferences after a warm-up and print
// the median and 95th percentile in milliseconds as JSON.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include "classifier.hpp"

namespace {

constexpr int kWarmup = 10;

int usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s MODEL.onnx IMAGE...\n"
               "       %s --bench N MODEL.onnx IMAGE\n",
               program, program);
  return 2;
}

// Value at quantile q of sorted values, by nearest rank.
double quantile(const std::vector<double>& sorted, double q) {
  auto rank = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1) + 0.5);
  return sorted[std::min(rank, sorted.size() - 1)];
}

int bench(int iterations, const std::string& model, const std::string& image_path) {
  cv::Mat gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (gray.empty()) {
    std::fprintf(stderr, "cannot read %s\n", image_path.c_str());
    return 1;
  }
  sdi::Classifier classifier(model);
  for (int i = 0; i < kWarmup; ++i) {
    classifier.probabilities(gray);
  }
  std::vector<double> ms;
  ms.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    classifier.probabilities(gray);
    std::chrono::duration<double, std::milli> took = std::chrono::steady_clock::now() - start;
    ms.push_back(took.count());
  }
  std::sort(ms.begin(), ms.end());
  std::printf(
      "{\"runtime\": \"OpenCV DNN (C++)\", \"opencv\": \"%s\", \"threads\": %d, "
      "\"iterations\": %d, \"median_ms\": %.3f, \"p95_ms\": %.3f}\n",
      CV_VERSION, cv::getNumThreads(), iterations, quantile(ms, 0.5), quantile(ms, 0.95));
  return 0;
}

int classify(const std::string& model, int count, char** images) {
  sdi::Classifier classifier(model);
  for (int i = 0; i < count; ++i) {
    cv::Mat gray = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
      std::fprintf(stderr, "cannot read %s\n", images[i]);
      return 1;
    }
    auto p = classifier.probabilities(gray);
    auto best = std::max_element(p.begin(), p.end()) - p.begin();
    std::printf("%s\t%s\t%.4f\n", images[i], sdi::kClasses[best].c_str(), p[best]);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
      if (argc != 5) {
        return usage(argv[0]);
      }
      int iterations = std::atoi(argv[2]);
      if (iterations < 1) {
        return usage(argv[0]);
      }
      return bench(iterations, argv[3], argv[4]);
    }
    if (argc < 3) {
      return usage(argv[0]);
    }
    return classify(argv[1], argc - 2, argv + 2);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
