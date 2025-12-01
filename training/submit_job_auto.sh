#!/bin/bash

# Define the range of n values for layer_target
start=1
end=50

# Define the base value for var
# !!! The Target should always be the first element of the list !!!
#var_list=('dp' 'temp' 'saln')    ## Used to predict the dp with temp and saln as predictors
#var_list=('dp')
#var_list=('v' 'u')
#var_list=('u' 'v')
#var_list=('saln')
#var_list=('dpmixl' 'temp' 'saln' 'dp')
#var_list=('ubavg')
#var_list=('vbavg')
#var_list=('pbavg')
#var_list=('ECO_flac' 'temp' 'saln' 'dp')
#var_list=('ECO_diac' 'temp' 'saln' 'dp')
#var_list=('ECO_cclc' 'temp' 'saln' 'dp')
#var_list=('ECO_fla' 'temp' 'saln' 'dp')
#var_list=('ECO_dia' 'temp' 'saln' 'dp')
#var_list=('ECO_ccl' 'temp' 'saln' 'dp')
#var_list=('ECO_fla')
#var_list=('ECO_dia')
#var_list=('ECO_ccl')
#var_list=('ECO_flac')
#var_list=('ECO_diac')
var_list=('ECO_cclc')
#var_list=('ECO_fla' 'ECO_flac')
#var_list=('ECO_dia' 'ECO_diac')
#var_list=('ECO_ccl' 'ECO_cclc')
#var_list=('ECO_sil')
#var_list=('ECO_oxy')
#var_list=('CO2_c')
#var_list=('temp')   ## Used to predict only one variable out of its neighbouring layers
var_target=${var_list[0]}

# Iterate through the range of n values
for n in $(seq $start $end); do
    # Determine the range for layer_pred
    if [[ "${var_list[0]}" == "ubavg" || "${var_list[0]}" == "vbavg" || "${var_list[0]}" == "pbavg" || "${var_list[0]}" == "dpmixl" ]]; then
        layer_pred_values=(1)
    else
        if [ $n -eq 1 ]; then
            layer_pred_values=(1 2)
        elif [ $n -eq 50 ]; then
            layer_pred_values=(49 50)
        else
            layer_pred_values=($((n-1)) $n $((n+1)))
	    #layer_pred_values=( $n )
        fi
    fi

    layer_pred=$(IFS=,; echo "${layer_pred_values[*]}")
    var_predictors=$(IFS=,; echo "${var_list[*]}")

    # Create a temporary copy of job_auto.sh
    temp_script="job_auto_temp_${n}.sh"
    cp job_auto.sh $temp_script

    # Use sed to replace the placeholders with the current values 
    sed -i "s/^var=.*/var='$var_predictors'/" $temp_script
    sed -i "s/^layer_pred=.*/layer_pred='$layer_pred'/" $temp_script
    sed -i "s/^layer_target=.*/layer_target='$n'/" $temp_script

    # Modify the output and error file names dynamically based on var, layer_pred, and layer_target
    sed -i "s|log/training_3V201_layer_pred_%A_layer_target_%a.o|log/training_${var_target}_layer_pred_${layer_pred}_layer_target_${n}.o|" $temp_script
    sed -i "s|log/training_3V201_layer_pred_%A_layer_target_%a.e|log/training_${var_target}_layer_pred_${layer_pred}_layer_target_${n}.e|" $temp_script

    # Submit the modified script
    echo "Submitting job with var=$var_list, layer_pred=$layer_pred, layer_target=$n"
    sbatch $temp_script

    # Clean up the temporary script
    rm $temp_script
done
