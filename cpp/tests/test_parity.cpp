// Compare the C++ preprocessing and inference with the probabilities that Python computed.
//
// Usage: test_parity FIXTURE_DIR, where the directory comes from `sdi parity-fixture`.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "classifier.hpp"

namespace {

constexpr double kTolerance = 1e-4;

// The numbers of the "probabilities" array; the fixture is written by our own code, so a full
// JSON parser is not needed.
std::vector<double> expected_probabilities(const std::string& path) {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  std::vector<double> values;
  auto key = text.find("\"probabilities\"");
  if (key == std::string::npos) return values;
  const char* cursor = text.c_str() + text.find('[', key) + 1;
  while (*cursor != ']' && *cursor != '\0') {
    char* end = nullptr;
    double value = std::strtod(cursor, &end);
    if (end == cursor) {
      ++cursor;
      continue;
    }
    values.push_back(value);
    cursor = end;
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const auto expected = expected_probabilities(dir + "/expected.json");
  sdi::Classifier classifier(dir + "/model.onnx");
  const auto actual =
      classifier.probabilities(cv::imread(dir + "/image.png", cv::IMREAD_GRAYSCALE));
  if (expected.size() != sdi::kClasses.size() || actual.size() != expected.size()) {
    std::fprintf(stderr, "expected %zu probabilities, got %zu from JSON and %zu from C++\n",
                 sdi::kClasses.size(), expected.size(), actual.size());
    return 1;
  }
  double worst = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    worst = std::max(worst, std::abs(actual[i] - expected[i]));
  }
  std::printf("largest difference %.2e (tolerance %.0e)\n", worst, kTolerance);
  return worst <= kTolerance ? 0 : 1;
}
