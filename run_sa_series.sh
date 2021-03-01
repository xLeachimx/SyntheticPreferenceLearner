#!/bin/bash
# MIN=$1
# MAX=$2
# CONFIG=$3
#
# for i in `seq $MIN $MAX`
# do
#   NEWFILE=$CONFIG\_results_$i\L_256
#   rm timing.dat
#   python3 wrapper.py -l $i -o $NEWFILE.csv $CONFIG
#   zip $NEWFILE.zip $NEWFILE.csv $CONFIG timing.dat
# done

# config_ary='baseline_multi five_113RPF five_227RPF five_CPNet_0 five_CPNet_7'
# single_config_ary='learn_min'
# single_config_ary='learn_aso'
config_ary='learn_five_13311aso learn_five_33722aso'

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_113RPF_util
  echo $config.config
  python3 wrapper.py -p 3 2 -i 113RPF_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_227RPF_util
  echo $config.config
  python3 wrapper.py -p 3 2 -i 227RPF_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done


for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_113RPF_mm
  echo $config.config
  python3 wrapper.py -p 3 3 -i 113RPF_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_227RPF_mm
  echo $config.config
  python3 wrapper.py -p 3 3 -i 227RPF_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

# for config in $single_config_ary
# do
#   rm timing.dat
#   OUTFILE=$config\_results_SA_113RPF_ASO
#   echo $config.config
#   python3 wrapper.py -p 3 1 -i 113RPF_learn.config -o $OUTFILE.csv $config.config
#   zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
# done
#
# for config in $single_config_ary
# do
#   rm timing.dat
#   OUTFILE=$config\_results_SA_227RPF_ASO
#   echo $config.config
#   python3 wrapper.py -p 3 1 -i 227RPF_learn.config -o $OUTFILE.csv $config.config
#   zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
# done
#
# for config in $single_config_ary
# do
#   rm timing.dat
#   OUTFILE=$config\_results_SA_33722ASO_ASO
#   echo $config.config
#   python3 wrapper.py -p 3 1 -i 33722ASO_learn.config -o $OUTFILE.csv $config.config
#   zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
# done
#
# for config in $single_config_ary
# do
#   rm timing.dat
#   OUTFILE=$config\_results_SA_13311ASO_ASO
#   echo $config.config
#   python3 wrapper.py -p 3 1 -i 13311ASO_learn.config -o $OUTFILE.csv $config.config
#   zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
# done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_13311ASO_util
  echo $config.config
  python3 wrapper.py -p 3 2 -i 13311ASO_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_33722ASO_util
  echo $config.config
  python3 wrapper.py -p 3 2 -i 33722ASO_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_13311ASO_mm
  echo $config.config
  python3 wrapper.py -p 3 3 -i 13311ASO_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

for config in $config_ary
do
  rm timing.dat
  OUTFILE=$config\_results_SA_33722ASO_mm
  echo $config.config
  python3 wrapper.py -p 3 3 -i 33722ASO_learn.config -o $OUTFILE.csv $config.config
  zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
done

# python3 wrapper.py -o five_227RPF.config_results_113RPF.csv
