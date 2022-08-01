#!/bin/bash
for (( i=0; i <= 124; i++ ))
do
    python single_model_test.py --class_id $i
done
