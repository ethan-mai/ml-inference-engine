#include "engine/engine.hpp"

#include <nlohmann/json.hpp>

namespace engine {

std::string version() {
    nlohmann::json j = {{"name", "ml_inference_engine"}, {"version", "0.0.1"}};
    return j.at("name").get<std::string>() + " " + j.at("version").get<std::string>();
}

}  // namespace engine
