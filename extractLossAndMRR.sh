#!/bin/bash

# activate bash checks
set -o nounset  # exit with error on unset variables
set -o errexit  # exit if any statement returns a non-true return value
set -o pipefail # exit if any pipe command is failing

for i in "$@"; do
  case $i in
    # tool parameter
    -f=*|--file=*)
      file="${i#*=}"
      shift # past argument=value
      ;;
    # default for unknown parameter
    -*|--*)
      echo "unknow option $i provided"
      exit 1
      ;;
    *)
      ;;
  esac
done

cp ./csv_template.csv ./csv_${file}.csv

outputFile="./csv_${file}.csv"
i=0
while read -r line
do 
    if [[ "${line}" == *"Valid MRR:"* ]]; then
        loss=$(echo ${line} | grep -o -P '(?<=Loss:).*(?=,)')
        mrr=$(echo ${line} | grep -oP '(?<=Valid MRR: ).*')
        echo "${i},${loss},${mrr}" >> ${outputFile}
        i=$(( $i+1 ))
    fi
done < "${file}"

echo "Done."

set +o nounset  # exit with error on unset variables
set +o errexit  # exit if any statement returns a non-true return value
set +o pipefail # exit if any pipe command is failing