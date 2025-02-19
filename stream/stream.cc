#include <nlohmann/json.hpp>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

using nlohmann::json;

enum ParslTaskKind {
  Shell,
  Python,
  Finish,
};

struct BinaryParslTaskArgs {
  std::size_t data_size;
  std::uint8_t data[];

  constexpr std::size_t size() const {
    return sizeof(BinaryParslTaskArgs) + data_size * sizeof(std::uint8_t);
  }
};

struct ParslTaskArgs {
  ParslTaskKind task_kind;
  std::vector<std::string> cmd;
  std::string input_file;
  std::string output_file;
  std::size_t request_mem_size;

  static ParslTaskArgs from_binary(BinaryParslTaskArgs const *binary_args);

  std::unique_ptr<BinaryParslTaskArgs> to_binary() const;
};

void to_json(json &j, ParslTaskArgs const &p) {
  j = json{{"task_kind", p.task_kind},
           {"cmd", p.cmd},
           {"input_file", p.input_file},
           {"output_file", p.output_file}};
}

void from_json(json const &j, ParslTaskArgs &p) {
  j.at("task_kind").get_to(p.task_kind);
  j.at("cmd").get_to(p.cmd);
  j.at("input_file").get_to(p.input_file);
  j.at("output_file").get_to(p.output_file);
}

ParslTaskArgs
    ParslTaskArgs::from_binary(BinaryParslTaskArgs const *binary_args) {
  std::vector<std::uint8_t> data(binary_args->data,
                                 binary_args->data + binary_args->data_size);
  json j = json::from_bson(data);

  printf("bson_json: %s\n", j.dump().c_str());

  return j.get<ParslTaskArgs>();
}

std::unique_ptr<BinaryParslTaskArgs> ParslTaskArgs::to_binary() const {
  json j = *this;
  auto data = json::to_bson(j);

  auto binary_task_args = (BinaryParslTaskArgs *)std::malloc(
      sizeof(BinaryParslTaskArgs) + data.size() * sizeof(std::uint8_t));

  binary_task_args->data_size = data.size();
  std::memcpy(
      binary_task_args->data, data.data(), data.size() * sizeof(std::uint8_t));

  return std::unique_ptr<BinaryParslTaskArgs>(binary_task_args);
}

/// 暂时以读取标准输入来模拟通过inotify读取parsl的文件操作，
/// 因此模拟时也可以利用管道来将parsl进程与legion进程串联
int last_file_size = 0;

std::optional<ParslTaskArgs> read_parsl_request() {
    std::ifstream input_file_stream;
    input_file_stream.open("./playground/task_args.json");
    input_file_stream.seekg(0, std::ios::end);
    
    std::streampos file_size = input_file_stream.tellg();
    printf("file_size: %d\n", static_cast<int>(file_size));
    if (file_size == last_file_size) {
        input_file_stream.close();
        return std::nullopt;
    }
    std::string input;

    int new_size = static_cast<int>(file_size);
    input_file_stream.seekg(-(new_size - last_file_size), std::ios::end);
    last_file_size = new_size;
    
    std::getline(input_file_stream, input);
    auto j = json::parse(input.c_str());
    auto parsl_task_args = j.get<ParslTaskArgs>();

    if (parsl_task_args.task_kind == ParslTaskKind::Finish) {
        printf("close listen input_file\n");
        input_file_stream.close();
        return parsl_task_args;
    }

    input_file_stream.close();
    return parsl_task_args;
}
int main() {
    
    while(true) {
        auto const parsl_task_args_opt = read_parsl_request();
        if (!parsl_task_args_opt.has_value()) {
            printf("sleep and wait\n");
            sleep(1);
            continue;
        } else {
            if (parsl_task_args_opt.value().task_kind == ParslTaskKind::Finish) {
                printf("finished\n");
                return 0;
            } else {
                printf("get result parsl_task_args_opt: %s\n", parsl_task_args_opt.value().input_file.c_str());
            }
        }
    }
    return 0;
}