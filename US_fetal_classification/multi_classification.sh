#!/bin/sh
for i in DenseNet_169 V&G_16 MobileNetV2
	do
		python classification.py Plane $i 
done