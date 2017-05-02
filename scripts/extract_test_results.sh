#!/bin/bash
if [[ -z ${2} ]]; then 
  echo "Usage: ${0} <versionfile> <basepath>"
  exit 1
fi

touch ./results.csv

while read line; do
  ls ${2}/${line} | sort -V > /tmp/ext_files
  echo '#################################################################'
  echo "                          -----$line-----"

  ./extract_runtime_data.py -f /tmp/ext_files -p ../w02/edge_based/output/${line}
  echo >> ./results.csv
  cat ../w02/edge_based/output/${line}/results.csv >> ./results.csv
done < ${1}
