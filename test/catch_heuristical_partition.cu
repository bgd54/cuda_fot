#include <algorithm>
#include <catch.hpp>
#include <set>
#include <sstream>

#include "details/heuristical_partition.hpp"

static Mesh generateMesh(const std::string &input) {
  std::stringstream mesh_input(input);
  Mesh mesh({&mesh_input}, {2});
  return mesh;
}

static GraphCSR<MY_SIZE> generateGraph(const std::string &input) {
  Mesh mesh = generateMesh(input);
  return GraphCSR<MY_SIZE>(mesh.numPoints(0), mesh.numCells(),
                           mesh.cell_to_node[0]);
}

TEST_CASE("GraphCSR: sanity check", "[graph CSR]") {
  auto graph = generateGraph("4 4\n"
                             "0 1\n"
                             "0 2\n"
                             "1 2\n"
                             "2 3\n");
  bool point_indices_increasing =
      std::is_sorted(graph.pointIndices(), graph.pointIndices() + 4);
  CHECK(point_indices_increasing);
  CHECK((sizeof(MY_SIZE)) == 4);
}

TEST_CASE("GraphCSR: reorderInverse", "[graph CSR]") {
  GraphCSR<MY_SIZE> graph = generateGraph("4 4\n"
                                          "0 1\n"
                                          "0 2\n"
                                          "1 2\n"
                                          "2 3\n");
  std::vector<std::vector<MY_SIZE>> expect = {{3}, {2, 3}, {1, 3}, {0, 1, 2}};
  graph.reorderInverse({3, 1, 0, 2});
  for (MY_SIZE i = 0; i < 4; ++i) {
    INFO("checking point" << i);
    const MY_SIZE start = graph.pointIndices()[i];
    const MY_SIZE end = graph.pointIndices()[i + 1];
    const MY_SIZE num_points = end - start;
    REQUIRE(num_points == (expect[i].size()));
    std::sort(graph.cellEndpoints() + start, graph.cellEndpoints() + end);
    bool same_endpoints = std::equal(expect[i].begin(), expect[i].end(),
                                     graph.cellEndpoints() + start);
    CHECK(same_endpoints);
  }
}

TEST_CASE("HeuristicalPartition: two small segments",
          "[heuristical partition]") {
  auto mesh = generateMesh("6 4\n"
                           "4 1\n"
                           "3 2\n"
                           "0 3\n"
                           "1 5\n");
  SECTION("one block per coarse block") {
    std::vector<MY_SIZE> partition = partitionOurs(mesh, 2, 2, 1.001);
    REQUIRE((partition.size()) == 4);
    std::vector<MY_SIZE> expected{0, 1, 1, 0};
    for (unsigned i = 0; i < 4; ++i) {
      INFO("at cell " << i);
      CHECK((partition[i]) == (expected[i]));
    }
  }
  SECTION("two blocks per coarse block") {
    std::vector<MY_SIZE> partition = partitionOurs(mesh, 1, 2, 1.001);
    std::set<MY_SIZE> block_set(partition.begin(), partition.end());
    CHECK((block_set.size()) == 4);
  }
}

TEST_CASE("HeuristicalPartition: trivial solution", "[heuristical partition]") {
  auto mesh = generateMesh("16 18\n"
                           "1 3\n"
                           "4 3\n"
                           "1 2\n"
                           "3 0\n"
                           "5 7\n"
                           "1 5\n"
                           "6 4\n"
                           "6 0\n"
                           "2 7\n"
                           "9 11\n"
                           "12 11\n"
                           "9 10\n"
                           "11 8\n"
                           "13 15\n"
                           "9 13\n"
                           "14 12\n"
                           "14 8\n"
                           "10 15\n");
  std::vector<MY_SIZE> partition = partitionOurs(mesh, 5, 9, 1.001);
  REQUIRE((partition.size()) == 18);
  std::vector<std::vector<MY_SIZE>> groups{
      {1, 3, 6, 7}, {2, 4, 5, 8}, {10, 12, 15, 16}, {11, 13, 14, 17}};
  for (unsigned i = 0; i < groups.size(); ++i) {
    INFO("in group " << i);
    for (unsigned j = 1; j < groups[i].size(); ++j) {
      CHECK((partition[groups[i][j]]) == (partition[groups[i][0]]));
    }
  }
}

/**
 * Works funny on the following graph:
 * auto mesh = generateMesh("12 14\n"              //  o-o    o-o   //
 *                          "1 3\n"                //  |/     |/    //
 *                          "4 3\n"                //  o      o     //
 *                          "1 2\n"                //  |      |     //
 *                          "3 0\n"                //  o      o     //
 *                          "2 5\n"                //  |\     |\    //
 *                          "5 1\n"                //  o-o    o-o   //
 *                          "4 0\n"                // Result:
 *                          "7 9\n"                //               //
 *                          "10 9\n"               //   1   1       //
 *                          "7 8\n"                //  /|\ /|\      //
 *                          "9 6\n"                // 0 | 0 | 1     //
 *                          "8 11\n"               //  \|/ \|/      //
 *                          "11 7\n"               //   0   0       //
 *                          "10 6\n");             //               //
 */
