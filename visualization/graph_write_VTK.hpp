#ifndef GRAPH_WRITE_VTK
#define GRAPH_WRITE_VTK
#include "geometry_routines.hpp"
#include "mesh.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

const size_t VTK_IND_THR_COL = 0;
const size_t VTK_IND_BLK_ID = 1;
const size_t VTK_IND_BLK_COL = 2;

// MeshDim -> cell type
const std::map<unsigned, unsigned> VTK_CELL_TYPES{
    {2, 3}, // line
    {4, 9}, // quad
    {8, 12} // hexahedron
};

void writeMeshToVTKAscii(
    std::string filename, const data_t &cell_list, const data_t &point_coords,
    const std::vector<std::vector<std::uint16_t>> &cell_colors) {
  assert(sizeof(MY_SIZE) == cell_list.getTypeSize());
  assert(sizeof(float) == point_coords.getTypeSize());
  std::ofstream fout(filename);
  if (!fout.good()) {
    std::cerr << "can't open file for write " << filename << "\n";
    exit(-1);
  }
  fout << "# vtk DataFile Version 2.0\n Output from graph partitioning test.\n";
  fout << "ASCII \nDATASET UNSTRUCTURED_GRID\n\n";
  fout << "POINTS " << point_coords.getSize() << " float\n";
  for (MY_SIZE i = 0; i < point_coords.getSize(); ++i) {
    fout << point_coords.operator[]<float>(i *point_coords.getDim() + 0) << " "
         << point_coords.operator[]<float>(i *point_coords.getDim() + 1) << " "
         << (point_coords.getDim() == 2
                 ? 0.0
                 : point_coords.operator[]<float>(i *point_coords.getDim() + 2))
         << "\n";
  }
  fout << "\nCELLS " << cell_list.getSize() << " "
       << (cell_list.getDim() + 1) * cell_list.getSize() << "\n";
  for (MY_SIZE i = 0; i < cell_list.getSize(); ++i) {
    fout << cell_list.getDim();
    if (cell_list.getDim() == 4) {
      std::array<float, 12> points;
      for (MY_SIZE j = 0; j < cell_list.getDim(); ++j) {
        std::copy_n(
            point_coords.cbegin<float>() +
                3 * cell_list.operator[]<MY_SIZE>(i *cell_list.getDim() + j),
            3, points.begin() + 3 * j);
      }
      std::array<MY_SIZE, 4> permutation = geom::two_d::reorderQuad(points);
      for (MY_SIZE j = 0; j < cell_list.getDim(); ++j) {
        fout << " "
             << cell_list.operator[]<MY_SIZE>(i *cell_list.getDim() +
                                              permutation[j]);
      }
    } else {
      for (MY_SIZE j = 0; j < cell_list.getDim(); ++j) {
        fout << " " << cell_list.operator[]<MY_SIZE>(i *cell_list.getDim() + j);
      }
    }
    fout << "\n";
  }
  fout << "\nCELL_TYPES " << cell_list.getSize() << "\n";
  assert(VTK_CELL_TYPES.count(cell_list.getDim()) == 1);
  for (MY_SIZE i = 0; i < cell_list.getSize(); ++i) {
    fout << VTK_CELL_TYPES.at(cell_list.getDim()) << "\n";
  }
  fout << "\n";

  if ((cell_colors.size() > VTK_IND_THR_COL &&
       cell_colors[VTK_IND_THR_COL].size() > 0) ||
      (cell_colors.size() > VTK_IND_BLK_ID &&
       cell_colors[VTK_IND_BLK_ID].size() > 0) ||
      (cell_colors.size() > VTK_IND_BLK_COL &&
       cell_colors[VTK_IND_BLK_COL].size() > 0)) {
    fout << "CELL_DATA " << cell_list.getSize() << "\n";
  }

  if (cell_colors.size() > VTK_IND_THR_COL &&
      cell_colors[VTK_IND_THR_COL].size() > 0) {
    fout << "SCALARS VTK_IND_THR_COL int 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < cell_colors[VTK_IND_THR_COL].size(); ++i) {
      fout << static_cast<int>(cell_colors[VTK_IND_THR_COL][i]) << "\n";
    }
  }
  fout << "\n";
  if (cell_colors.size() > VTK_IND_BLK_ID &&
      cell_colors[VTK_IND_BLK_ID].size() > 0) {
    fout << "SCALARS VTK_IND_BLK_ID int 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < cell_colors[VTK_IND_BLK_ID].size(); ++i) {
      fout << static_cast<int>(cell_colors[VTK_IND_BLK_ID][i]) << "\n";
    }
  }
  fout << "\n";
  if (cell_colors.size() > VTK_IND_BLK_COL &&
      cell_colors[VTK_IND_BLK_COL].size() > 0) {
    fout << "SCALARS VTK_IND_BLK_COL int 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < cell_colors[VTK_IND_BLK_COL].size(); ++i) {
      fout << static_cast<int>(cell_colors[VTK_IND_BLK_COL][i]) << "\n";
    }
  }
  fout.close();
}
#endif /* end of include guard: GRAPH_WRITE_VTK*/
