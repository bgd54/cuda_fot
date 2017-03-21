#!/usr/bin/python3

import os
import argparse

parser = argparse.ArgumentParser(description=\
        'Parse OP2 outputs and print data to .csv files.')
parser.add_argument('--version_file', '-f', default='versions',\
        help='name of the file containing version filenames' + \
            'which you want to process.')
parser.add_argument('--path', '-p', default='',nargs=1,\
        help='Path where test results placed.' )
parser.add_argument('--print_bw', '--bw', action='store_true',default=False,\
        help='Print bandwith data.' ) #TODO inkabb ide is fajlt megadni?
                                      #TODO jok-e legyeneke meg? stb..

args = parser.parse_args()


# Amit parseolni akarok:
# 
# count   plan time     MPI time(std)        time(std)           GB/s      GB/s   kernel name 
#  -------------------------------------------------------------------------------------------
#    1000;    0.0000;    0.0000(  0.0000);    1.2972(  0.0000);  142.0901;         ;   save_soln 
#    2000;    0.6823;    0.0000(  0.0000);    4.5262(  0.0000); 112.3703; 114.5890;   adt_calc 
#    2000;    0.6410;    0.0000(  0.0000);   16.1223(  0.0000);    0.0000;         ;   res_calc 
#    2000;    0.0103;    0.0000(  0.0000);    0.0957(  0.0000);    0.0000;         ;   bres_calc 
#    2000;    0.0000;    0.0000(  0.0000);    6.7230(  0.0000);  116.5117;         ;   update 
# Total plan time:   1.3336
# Max total runtime = 28.771502
 

class kernel:
    count = 0
    name = ""
    plantime=0.0
    time=0.0
    bandwith = None
    useful_bandwith = None
    MPI_time = 0.0

    def __init__(self,line,time=None, ptime=None):
      if time is None:
        line = line.strip().split(";")
        self.name = line[-1].strip()
        count =  int(line[0].strip())
        self.plantime = float(line[1].strip())
        self.MPI_time = float(line[2][0:line[2].find("(")].strip())
        self.time = float(line[3][0:line[3].find("(")].strip())
        self.bandwith = float(line[4].strip())
        self.useful_bandwith = \
                self.bandwith if line[5].strip() == "" else\
                float(line[5].strip())
      else:
        self.name=line
        self.time=time
        self.plantime=ptime

    def getTotalTime(self):
      return self.time-self.plantime





def processkernels(fname="cuda-AOS-2.8m"):
    kernels = []
    with open(fname,'r') as fin:
        line = ""
        while line.strip() != "-"*91:
            line = fin.readline()
        line = fin.readline()
        while not line.startswith("Total"):
            kernels.append(kernel(line))
            line = fin.readline()
        ptime = float(line.split(":")[-1].strip())
        time = float(fin.readline().split("=")[-1].strip())
        kernels.append( kernel("total", time, ptime))

    
    #kernelnames = kernels[0].name
    #kerneltimes = str(kernels[0].time-kernels[0].plantime)
    
    #for i in range(1,len(kernels)):
    #    kernelnames+=";"+kernels[i].name
    #    kerneltimes+=";"+str(kernels[i].time-kernels[i].plantime)
    #print(kernelnames)
    #print(kerneltimes)
    
    return kernels

def printtoCSV(results):
  with open("results.csv","w") as fout:
    line = ";".join([ ver for ver in results.keys()])
    fout.write(";"+line+"\n")
    kernel_names = [k.name for k in results[list(results.keys())[0]]]
    for i in range(len(kernel_names)):
      line = kernel_names[i]
      for ver in results.keys():
        line += ";%5lf"%results[ver][i].getTotalTime()
      fout.write(line+"\n")

def print_bw():
    print('TODO implement bw printing')

def collecttestresults(tests="tests", path="", printbw=False): #TODO bandwithek kiszedese MPI time szamolas
  results = dict()
  with open(tests, 'r') as fin:
    for test in fin:
        test = test.strip()
        testfilepath = os.path.join(path,test)
        print(testfilepath)
        currResults = processkernels(testfilepath)
        results[test] = currResults
  printtoCSV(results)

print(args)
collecttestresults(args.version_file, args.path, args.print_bw)

