#ifndef ARG_HPP
#define ARG_HPP

struct arg{
  //set_size: #elements, set_dim: #value per element, data_size: sizeof(datattype)
  int set_size, set_dim, data_size;
  //data pointers
  char *data, *data_d;

  //constructors
  arg(): set_size(0), set_dim(0), data_size(0),
    data(nullptr), data_d(nullptr){}
  arg(int, int, int, char*);
  
  //setting a new data
  void set_data(int, int, int, char*);
  
  //functions to manage state between host and device memory
  void update();
  void updateD();
  
  //dtor
  ~arg();
};

#endif /* end of guard ARG_HPP */
