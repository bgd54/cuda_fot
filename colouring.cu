#include "colouring.hpp"
#include "reorder.hpp"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <map>
#include <vector>

using adj_list_t = std::vector<std::vector<MY_SIZE>>;
using colourset_t = Mesh::colourset_t;

static adj_list_t getCellToCell(const std::vector<MY_SIZE> &mapping,
                                MY_SIZE mapping_dim) {
  const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
      GraphCSR<MY_SIZE>::getPointToCell(mapping.begin(), mapping.end(),
                                        mapping_dim);
  const MY_SIZE num_cells = mapping.size() / mapping_dim;
  adj_list_t cell_to_cell(num_cells);
  for (MY_SIZE i = 0; i < num_cells; ++i) {
    for (MY_SIZE offset = 0; offset < mapping_dim; ++offset) {
      const MY_SIZE point = mapping[mapping_dim * i + offset];
      const auto cell_range = point_to_cell.equal_range(point);
      for (auto it = cell_range.first; it != cell_range.second; ++it) {
        const MY_SIZE other_cell = it->second;
        if (other_cell != i) {
          cell_to_cell[i].push_back(other_cell);
        }
      }
    }
  }
  return cell_to_cell;
}

static std::vector<MY_SIZE> getCellOrder(adj_list_t &cell_to_cell) {
  std::vector<MY_SIZE> order;
  const MY_SIZE num_cells = cell_to_cell.size();
  std::vector<bool> processed(num_cells, false);
  std::vector<MY_SIZE> degrees(num_cells);
  for (MY_SIZE i = 0; i < num_cells; ++i) {
    degrees[i] = cell_to_cell[i].size();
  }
  for (MY_SIZE i = 0; i < num_cells; ++i) {
    const MY_SIZE v = [&]() {
      MY_SIZE mi = num_cells;
      for (MY_SIZE j = 0; j < num_cells; ++j) {
        if (processed[j]) {
          continue;
        }
        if (mi == num_cells || degrees[mi] >= degrees[j]) {
          mi = j;
        }
      }
      return mi;
    }();
    processed[v] = true;
    order.push_back(v);
    for (MY_SIZE n : cell_to_cell[v]) {
      const auto it =
          std::find(cell_to_cell[n].begin(), cell_to_cell[n].end(), v);
      assert(it != cell_to_cell[n].end());
      std::swap(*it, cell_to_cell[n].back());
      --degrees[n];
    }
  }
  return order;
}

static colourset_t
gatherOccupiedColours(const std::vector<MY_SIZE> &neighbours,
                      const std::vector<colourset_t> &cell_colours) {
  colourset_t occupied_colours{};
  for (MY_SIZE n : neighbours) {
    occupied_colours |= cell_colours[n];
  }
  return occupied_colours;
}

namespace {
struct ColouringPlan {
  std::vector<std::uint8_t> cell_colours;
  std::uint8_t num_cell_colours;
};
}

static ColouringPlan colourCellToCell(const adj_list_t &cell_to_cell,
                                      const std::vector<MY_SIZE> &order) {
  const MY_SIZE num_cells = cell_to_cell.size();
  std::vector<colourset_t> cell_colours(num_cells);
  colourset_t used_colours{};
  ColouringPlan colouring_plan{};
  colouring_plan.cell_colours.resize(num_cells);
  std::vector<MY_SIZE> set_sizes {};
  for (MY_SIZE i = num_cells - 1; i < num_cells; --i) {
    const MY_SIZE v = order[i];
    assert(cell_colours[v].none());
    const colourset_t occupied_colours =
        gatherOccupiedColours(cell_to_cell[v], cell_colours);
    colourset_t available_colours = ~occupied_colours & used_colours;
    if (available_colours.none()) {
      used_colours.set(colouring_plan.num_cell_colours++);
	  set_sizes.push_back(0);
      assert(colouring_plan.num_cell_colours <= used_colours.size());
      available_colours = ~occupied_colours & used_colours;
      assert(available_colours.any());
    }
    std::uint8_t colour =
        Mesh::template getAvailableColour<false>(available_colours, set_sizes);
    colouring_plan.cell_colours[v] = colour;
    colourset_t colourset{};
    colourset.set(colour);
    cell_colours[v] = colourset;
  }
  return colouring_plan;
}

std::pair<std::vector<std::uint8_t>, std::uint8_t>
colourCellsWithOrdering(const std::vector<MY_SIZE> &mapping,
                        MY_SIZE mapping_dim) {
  adj_list_t cell_to_cell = getCellToCell(mapping, mapping_dim);
  const std::vector<MY_SIZE> order = getCellOrder(cell_to_cell);
  ColouringPlan colouring_plan = colourCellToCell(cell_to_cell, order);
  return std::make_pair(colouring_plan.cell_colours,
                        colouring_plan.num_cell_colours);
}
