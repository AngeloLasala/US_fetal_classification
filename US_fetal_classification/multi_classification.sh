#!/bin/sh
for i in DenseNet_169 VGG_16 MobileNetV2
	do
		python classification.py Plane $i 
done