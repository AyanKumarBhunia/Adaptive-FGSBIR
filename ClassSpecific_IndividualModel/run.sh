#!/bin/bash
for (( i=83; i <= 124; i++ ))
do
    python main.py --class_id $i
done