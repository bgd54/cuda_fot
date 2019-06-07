import re

def get_runtime_from_op2_output(output):
#      count   plan time     MPI time(std)        time(std)           GB/s      GB/s   kernel name
#   -------------------------------------------------------------------------------------------
#     1000;    0.0000;    0.0000(  0.0000);    0.5591(  0.0000);  494.5006;         ;   save_soln
#     2000;    1.3487;    0.0000(  0.0000);    2.3646(  0.0000); 187.5255; 228.2072;   adt_calc
#     2000;    1.9029;    0.0000(  0.0000);    5.3852(  0.0000);    0.0000;         ;   res_calc
#     2000;    0.0419;    0.0000(  0.0000);    0.0969(  0.0000);    0.0000;         ;   bres_calc
#     2000;    0.0000;    0.0000(  0.0000);    2.2864(  0.0000);  423.2201;         ;   update
#  Total plan time:   3.2935
#  Max total runtime = 10.696117
  idx = 0
  while idx != len(output):
    if output[idx].strip().startswith('-------------------------------------------------------------------------------------------'):
      break
    idx +=1
  idx += 1
  header = []
  runtimes = []
  while not output[idx].startswith("Total"):
    row = output[idx].strip().split(';')
    plan = float(row[1].strip())
    rt = float(row[3][:row[3].index('(')])
    runtimes.append("{:.4f}".format(rt-plan))
    header.append(row[-1].strip())
    idx += 1

  return header, runtimes



def parse_multiple_runs(filename):
  lines = []
  with open(filename, 'r') as fin:
    lines = fin.read().split('\n')

  separator = re.compile("^_+(.*[^_])_+$")
  header = []
  for idx in range(len(lines)):
    match = separator.match(lines[idx])
    if match is not None:
      h, rt = get_runtime_from_op2_output(lines[idx:])
      headerstr = '\t;' + ';'.join(h)
      if h != header:
        print(headerstr)
        header = h
      print(match.group(1)+';'+';'.join(rt))

