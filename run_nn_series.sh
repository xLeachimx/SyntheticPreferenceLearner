MIN=$1
MAX=$2
# config_ary='baseline_multi five_113RPF five_227RPF five_CPNet_0 five_CPNet_7'
config_ary='learn_new'

for config in $config_ary
do
  for i in `seq $MIN $MAX`
  do
    OUTFILE=$config\_results_$i\L_256
    echo $OUTFILE
    rm timing.dat
    python3 wrapper.py -l $i -o $OUTFILE.csv $config.config
    zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
  done
done
