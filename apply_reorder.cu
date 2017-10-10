#include "problem.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>

constexpr float PARTITION_TOLERANCE = 1.001;

enum ArgName { PRINT_HELP, INPUT, GPS, METIS, _ARG_NAME_LEN };
using args_t = std::array<std::string, _ARG_NAME_LEN>;

args_t getArguments(int argc, char **argv) {
  args_t arguments;
  arguments.fill("");
  if (argc < 3) {
    arguments[PRINT_HELP] = true;
    return arguments;
  }
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      arguments.fill("");
      arguments[PRINT_HELP] = "true";
      return arguments;
    } else if (std::strcmp(argv[i], "--metis") == 0) {
      if (i == argc - 1) {
        std::cerr << "No file output found for --metis." << std::endl;
        arguments[PRINT_HELP] = "true";
        return arguments;
      }
      arguments[METIS] = argv[++i];
    } else if (std::strcmp(argv[i], "--gps") == 0) {
      if (i == argc - 1) {
        std::cerr << "No file output found for --gps." << std::endl;
        arguments[PRINT_HELP] = "true";
        return arguments;
      }
      arguments[GPS] = argv[++i];
    } else {
      if (arguments[INPUT].empty()) {
        arguments[INPUT] = argv[i];
      } else if (arguments[GPS].empty()) {
        arguments[GPS] = argv[i];
      }
    }
  }
  return arguments;
}

void printUsage(const char *prog_name) {
  std::cerr << "Usage: " << prog_name
            << " in_file [--gps <gps_out_file>] [--metis <metis_out_file>]"
            << std::endl;
  std::cerr << std::endl
            << "--gps <gps_out_file>            writes GPS reordered mesh "
               "to the file\n"
               "--metis <metis_out_file>        writes METIS partitioned mesh "
               "to the file\n"
               "                                writes partition to the file "
               "with suffix\n"
               "                                '_part'"
            << std::endl;
}

int main(int argc, char **argv) {
  args_t arguments = getArguments(argc, argv);
  if (!arguments[PRINT_HELP].empty()) {
    printUsage(argv[0]);
    return 1;
  }
  try {
    std::ifstream f(argv[1]);
    if (!f) {
      std::cerr << "Error opening input file: " << argv[1] << std::endl;
      return 3;
    }
    Problem<> problem(f, 288);
    problem.reorder();
    if (!arguments[GPS].empty()) {
      std::ofstream f_gps_out(arguments[GPS]);
      problem.mesh.writeCellList(f_gps_out);
    }
    if (!arguments[METIS].empty()) {
      std::ofstream f_metis_out(arguments[METIS]);
      std::ofstream f_metis_out_part(arguments[METIS] + "_part");
      problem.partition(PARTITION_TOLERANCE);
      problem.reorderToPartition();
      problem.renumberPoints();
      problem.mesh.writeCellList(f_metis_out);
      problem.writePartition(f_metis_out_part);
    }
  } catch (ScotchError &e) {
    std::cerr << "Error during Scotch reordering: " << e.errorCode << std::endl;
    return 2;
  } catch (InvalidInputFile &e) {
    std::cerr << "Error in input file (" << e.input_type
              << ", line = " << e.line << ")" << std::endl;
    return 4;
  }
  return 0;
}
