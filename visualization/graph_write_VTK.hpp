#ifndef GRAPH_WRITE_VTK
#define GRAPH_WRITE_VTK
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "data_t.hpp"
#include <vector>

const size_t VTK_IND_THR_COL = 0;
const size_t VTK_IND_BLK_ID  = 1;
const size_t VTK_IND_BLK_COL = 2;

void writeGraphToVTKAscii(std::string filename, const data_t<float> &point_coords,
    const data_t<MY_SIZE> &edge_list, const std::vector<std::vector<std::uint16_t>> &edge_colors){

  std::ofstream fout(filename);
  if(!fout.good()){
    std::cerr << "can't open file for write " << filename << "\n";
    exit(-1);
  } 
  fout<< "# vtk DataFile Version 2.0\n Output from graph partitioning test.\n";
  fout<< "ASCII \nDATASET UNSTRUCTURED_GRID\n\n";
  fout<< "POINTS " << point_coords.getSize() << " float\n";
  for(MY_SIZE i=0; i<point_coords.getSize(); ++i){
    fout << point_coords[i*point_coords.getDim()+0] << " "
         << point_coords[i*point_coords.getDim()+1] << " "
         << 0.0 << "\n";
  }
  fout << "\nCELLS " << edge_list.getSize() << " " 
    << (edge_list.getDim()+1) * edge_list.getSize() << "\n";
  for(MY_SIZE i=0; i < edge_list.getSize(); ++i){
    fout << 2 << " " << edge_list[i*edge_list.getDim() + 0] << " "
      << edge_list[i * edge_list.getDim() +1] << "\n";
  }
  fout <<"\nCELL_TYPES " << edge_list.getSize() << "\n";
  for(MY_SIZE i=0; i < edge_list.getSize(); ++i){
    fout << "3\n"; //cell type 3 for edges
  }
  fout << "\n";
  
  if((edge_colors.size() > VTK_IND_THR_COL && edge_colors[VTK_IND_THR_COL].size() > 0)
      ||(edge_colors.size() > VTK_IND_BLK_ID && edge_colors[VTK_IND_BLK_ID].size() > 0)
      ||(edge_colors.size() > VTK_IND_BLK_COL && edge_colors[VTK_IND_BLK_COL].size() > 0)){
    fout << "CELL_DATA " << edge_list.getSize() << "\n";
  }
  
  if(edge_colors.size() > VTK_IND_THR_COL && edge_colors[VTK_IND_THR_COL].size() > 0){
    fout << "SCALARS VTK_IND_THR_COL int 1\nLOOKUP_TABLE default\n";
    for(size_t i=0; i<edge_colors[VTK_IND_THR_COL].size(); ++i){
      fout << static_cast<int>(edge_colors[VTK_IND_THR_COL][i])<<"\n";
    }
  }
  fout << "\n";
  if(edge_colors.size() > VTK_IND_BLK_ID && edge_colors[VTK_IND_BLK_ID].size() > 0){
    fout << "SCALARS VTK_IND_BLK_ID int 1\nLOOKUP_TABLE default\n";
    for(size_t i=0; i<edge_colors[VTK_IND_BLK_ID].size(); ++i){
      fout << static_cast<int>(edge_colors[VTK_IND_BLK_ID][i])<<"\n";
    }
  }
  fout << "\n";
  if(edge_colors.size() > VTK_IND_BLK_COL && edge_colors[VTK_IND_BLK_COL].size() > 0){
    fout << "SCALARS VTK_IND_BLK_COL int 1\nLOOKUP_TABLE default\n";
    for(size_t i=0; i<edge_colors[VTK_IND_BLK_COL].size(); ++i){
      fout << static_cast<int>(edge_colors[VTK_IND_BLK_COL][i])<<"\n";
    }
  }
  fout.close();
}
#endif /* end of include gurad: GRAPH_WRITE_VTK*/
