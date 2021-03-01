MIN=$1
MAX=$2
# config_ary='baseline_multi five_113RPF five_227RPF five_CPNet_0 five_CPNet_7'
# config_ary='learn_min'
# config_ary='portfolio_full portfolio_simple portfolio_similar portfolio_full_bal'
# config_ary='portfolio_similar_small  portfolio_simple_small'
# config_ary='portfolio_full_small'
config_ary='portfolio_full_bal_small'
# config_ary='portfolio_similar portfolio_full portfolio_full_bal'
# config_ary='portfolio_test'
# config_ary='portfolio_full_bal'

for config in $config_ary
do
  for i in `seq $MIN $MAX`
  do
    OUTFILE=$config\_gcn_3L_tweaked_results_full_examples_$i\L_256
    echo $OUTFILE
    rm timing.dat
    python3 wrapper.py -p 4 7 -l $i -o $OUTFILE.csv $config.config
    zip $OUTFILE.zip $OUTFILE.csv $config.config timing.dat
  done
done
