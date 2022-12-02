#!/bin/sh
for i in 0 2 4 6 8 -1
	do
		python classification.py Plane VGG_16 -frozen=$i
done