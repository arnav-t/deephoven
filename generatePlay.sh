#!/bin/bash

python3 generate.py
csvmidi out.csv out.mid
rm out.csv
timidity out.mid 