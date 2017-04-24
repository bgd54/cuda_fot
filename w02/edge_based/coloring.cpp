#include "coloring.hpp"
#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <string.h>


////////////////////////////////////////////////////////////////////////////////
//                              Global Coloring
////////////////////////////////////////////////////////////////////////////////

Coloring::Coloring(): color_reord(nullptr), arg_color_reord(),
              color_offsets(nullptr), colornum(0) {}

Coloring::Coloring(vector<vector<int> > _cr, vector<int> _coff, int nedge){
  colornum=_cr.size();
  color_offsets= (int*) malloc(colornum*sizeof(int));
  #pragma omp parallel for
  for(size_t i=0; i<_coff.size();++i){
    color_offsets[i]=_coff[i];
  }
  color_reord=(int*) malloc(nedge*sizeof(int));
  #pragma omp parallel for
  for(size_t i=0;i<_cr.size();++i){
    for(size_t j=0;j<_cr[i].size();j++){
      color_reord[j+(i>0?color_offsets[i-1]:0)]=_cr[i][j];
    }
  }
  arg_color_reord.set_data(nedge,1,sizeof(int),(char*)color_reord);
}

void Coloring::print(){
  printf("num of colors: %d\n",colornum);
  for(int i=0;i<colornum;++i){
    printf("color #%d #edge with this color %d:\n\t",i,color_offsets[i]);
    int start = i==0?0:color_offsets[i-1];
    for(int j=start; j<color_offsets[i];++j){
      printf("%3d ",color_reord[j]);
    }
    printf("\n");
  }
}

Coloring& Coloring::operator=(const Coloring& c){
  if(this == &c) return *this;
  free(color_offsets); 
  //color_reord handled in arg.

  colornum = c.colornum;
  color_offsets= (int*) malloc(colornum*sizeof(int));
  for(int i=0; i<colornum;++i) color_offsets[i]=c.color_offsets[i];

  color_reord=(int*) malloc(color_offsets[colornum-1]*sizeof(int));
  for(int  i=0; i<color_offsets[colornum-1];++i) color_reord[i]=c.color_reord[i];
  
  arg_color_reord.set_data( color_offsets[colornum-1], 1, sizeof(int),
      (char*)color_reord );
  
  return *this;
}

Coloring::Coloring(const Coloring& c) {
  if(this == &c) return;
  colornum = c.colornum;
  color_offsets= (int*) malloc(colornum*sizeof(int));
  for(int i=0; i<colornum;++i) color_offsets[i]=c.color_offsets[i];

  color_reord=(int*) malloc(color_offsets[colornum-1]*sizeof(int));
  for(int  i=0; i<color_offsets[colornum-1];++i) color_reord[i]=c.color_reord[i];
  arg_color_reord.set_data( color_offsets[colornum-1], 1, sizeof(int),
      (char*)color_reord );
}


Coloring::~Coloring(){
  //free(color_reord); //arg_color_reord frees color_reord
  free(color_offsets);
}



Coloring global_coloring(int* enode, int nedge, int nnode){

  vector<vector<int> > color_reord;
  vector<int> nodetocolor(
      nnode!=0? nnode:(*std::max_element(enode,enode+nedge)+1), 0 );

  for(int i=0;i<nedge;++i){
    size_t col= nodetocolor[enode[2*i+1]];//nodetocolor[enode[2*i+1]].size();

    if(col < color_reord.size()){
      color_reord[col].push_back(i);
      nodetocolor[enode[2*i+1]]++;//nodetocolor[enode[2*i+1]].push_back(col);
    } else if ( color_reord.size() == col ){
      nodetocolor[enode[2*i+1]]++;//nodetocolor[enode[2*i+1]].push_back(col);
      color_reord.push_back(vector<int>(1,i));
    } else {
      printf(
          "ERROR: col: %d greater than it could be:%d at ind: %d, write: %d\n",
          (int)col, (int)color_reord.size(), i, enode[2*i+1]);
    }
  }
  vector<int> offsets(color_reord.size(),color_reord[0].size());
  
  for(size_t i=1; i<color_reord.size(); ++i){
    offsets[i] = offsets[i-1] + color_reord[i].size();
  }
  return Coloring(color_reord,offsets, nedge);

}

////////////////////////////////////////////////////////////////////////////////
//                                Block Coloring
////////////////////////////////////////////////////////////////////////////////
Block_coloring::Block_coloring(): numblock(0), bs(0), nedge(0), colornum(nullptr),
  color_offsets(nullptr), color_reord(nullptr), reordcolor(nullptr){}

Block_coloring::Block_coloring(int _numblock, int _bs, int *_cnum,
    vector<vector<int> > _coff, int * _reord, int _nedge): 
  numblock(_numblock), bs(_bs), nedge(_nedge),colornum(_cnum){
  
  color_offsets = (int**) malloc(numblock*sizeof(int*));
  #pragma omp parallel for
  for(int bIdx=0;bIdx<numblock;++bIdx){
    color_offsets[bIdx] = (int*) malloc(_coff[bIdx].size()*sizeof(int));
    for(size_t i=0;i<_coff[bIdx].size();++i){
      color_offsets[bIdx][i] = _coff[bIdx][i];
    }
  }

  color_reord = _reord;

  reordcolor = (int*) malloc(nedge*sizeof(int));
  #pragma omp parallel for
  for(int bIdx=0; bIdx<numblock;++bIdx){
    int start= bIdx*bs;
    int end= std::min((bIdx+1)*bs,nedge);
    for(int reordIdx=start; reordIdx<end; reordIdx++){
      for(int col=0; col<colornum[bIdx];++col){
        if(color_offsets[bIdx][col]>reordIdx){
          reordcolor[reordIdx] = col;
          break;
        }
      }
    }
  }
  arg_colornum.set_data(numblock, 1, sizeof(int), (char*) colornum);
  arg_reordcolor.set_data(nedge, 1, sizeof(int), (char*) reordcolor);
  arg_color_reord.set_data(nedge, 1, sizeof(int), (char*) color_reord);

}

Block_coloring& Block_coloring::operator= (const Block_coloring& bc){
  if(this == &bc) return *this;
  for(int i=0; i<numblock;++i) free(color_offsets[i]);
  free(color_offsets);

  numblock = bc.numblock;
  bs = bc.bs;
  nedge = bc.nedge;

  colornum = (int*) malloc(nedge*sizeof(int));
  memcpy((void*) colornum, (void*)bc.colornum, nedge*sizeof(int) );

  color_reord = (int*) malloc(nedge*sizeof(int));
  memcpy((void*) color_reord, (void*)bc.color_reord, nedge*sizeof(int) );

  reordcolor = (int*) malloc(nedge*sizeof(int));
  memcpy((void*) reordcolor, (void*)bc.reordcolor, nedge*sizeof(int) );

  color_offsets = (int**) malloc(numblock*sizeof(int*));
  #pragma omp parallel for
  for(int bIdx=0;bIdx<numblock;++bIdx){
    color_offsets[bIdx] = (int*) malloc(colornum[bIdx]*sizeof(int));
    memcpy((void*) color_offsets[bIdx], (void*)bc.color_offsets[bIdx],
        colornum[bIdx]*sizeof(int) );
  }
  arg_colornum.set_data(numblock,1,sizeof(int),(char*)colornum);
  arg_reordcolor.set_data(nedge,1,sizeof(int),(char*)reordcolor);
  arg_color_reord.set_data(nedge,1,sizeof(int),(char*)color_reord);

  return *this;
}

Coloring Block_coloring::color_blocks(int* enode, int nedge){
  
  vector<vector<int> > block_reord;
  vector<set<int> > col_write;
  for(int bIdx=0;bIdx<numblock;++bIdx){
    size_t col=0;
    vector<int> writeByBlock;
    int start=bs*bIdx;
    int end = std::min(bs*(bIdx+1),nedge);
    for(int i=start;i<end;++i) 
      writeByBlock.push_back(enode[2*color_reord[i]+1]);
    
    //printf("bidx:%d/%d\n",bIdx,numblock);
    bool intersect=true; 
    while(col<block_reord.size()){
      intersect=false;
      for(size_t i=0;i<writeByBlock.size();++i){
        if(col_write[col].find(writeByBlock[i])!=col_write[col].end()){
          col++;
          //printf("%d %d\n",col, block_reord.size());
          intersect=true;
          break;
        }
      }
      if(!intersect){
        break;
      }
    }
    //printf("correct_col calculated:%d\n",col);
    if(col < block_reord.size()){
      block_reord[col].push_back(bIdx);
      for(int i:writeByBlock) col_write[col].insert(i);
    } else {
      block_reord.push_back(vector<int>(1,bIdx));
      col_write.push_back(set<int>(writeByBlock.begin(),writeByBlock.end()));
    }
  }
  //printf("numb:%d, start calc offsets\n",numblock);
  vector<int> offsets(block_reord.size(), block_reord[0].size());
  for(size_t i=1; i<block_reord.size();++i){
    offsets[i] = offsets[i-1]+block_reord[i].size();
  }

  return Coloring(block_reord,offsets,numblock);
}


Block_coloring::~Block_coloring(){
  printf("free memory used for coloring\n");
  //free(colornum); deallocated in arg_colornum
  for(int i=0; i<numblock;++i) free(color_offsets[i]);
  free(color_offsets);
  //free(color_reord); deallocated in arg_color_reord
 // free(reordcolor); deallocated in arg_reordcolor
}



Block_coloring block_coloring(int* enode, int nedge, int blockSize){
  int bs = blockSize;
  int numblock = (nedge-1)/bs+1;
  int* colornum = (int*) malloc(numblock*sizeof(int));
  vector<vector<int> > offsets(numblock);
  int* reord = (int*) malloc(nedge*sizeof(int));

  #pragma omp parallel for
  for(int bIdx=0;bIdx<numblock;++bIdx){
    //process 1 block
    int start = bIdx*bs, end= std::min((bIdx+1)*bs,nedge);
    vector<vector<int> > color2edge;
    vector<int> node_used;//TODO ehelyett egy map? 
    //process edges in block #bIdx
    for(int edgeIdx=start;edgeIdx<end;++edgeIdx){
      //printf("edge: %d\n", edgeIdx);
      int nodetowrite=enode[2*edgeIdx+1];
      size_t col= std::count(node_used.begin(), node_used.end(),nodetowrite);
      node_used.push_back(nodetowrite);

      if(col<color2edge.size()){
        color2edge[col].push_back(edgeIdx);
      } else if( col == color2edge.size()){
        color2edge.push_back(vector<int>(1,edgeIdx));
      } else {
        printf(
          "ERROR: col: %d greater than it could be:%d at ind: %d, write: %d\n",
          (int)col, (int)color2edge.size(), edgeIdx, nodetowrite);
      }
    }
    colornum[bIdx] = color2edge.size();
    int reord_pos=start;
    for(size_t col=0; col<color2edge.size();++col){
      for(size_t i=0; i<color2edge[col].size();++i){
        int edgeIdx = color2edge[col][i];
        reord[reord_pos]=edgeIdx;
        reord_pos++;
      }
      offsets[bIdx].push_back(reord_pos); //global offset.. local:reord_pos-start
    }
  }
  return Block_coloring(numblock, bs, colornum, offsets, reord, nedge);

}
