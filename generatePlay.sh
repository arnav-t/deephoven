#!/bin/bash

python3 generate.py
csvmidi out.csv out.midi 
rm out.csv
timidity out.mid 