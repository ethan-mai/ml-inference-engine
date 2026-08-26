#pragma once

#include <string>

namespace engine {

// Placeholder round-trip through nlohmann::json, proving the FetchContent'd
// dependency resolves at both compile and link time. Replaced by real IR
// parsing (Node, graph.json) as Week 1 progresses.
std::string version();

}  // namespace engine
