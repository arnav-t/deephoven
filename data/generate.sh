#!/bin/bash

convert () {
	midicsv $1 temp.csv
	perl -pi -e 's/[^[:ascii:]]//g' temp.csv
	python3 process.py temp.csv
	rm temp.csv
}

rm data.pt
find ./raw -name '*.mid' | while read midi; do
	convert $midi
done
