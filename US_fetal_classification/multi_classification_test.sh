#!/bin/sh
for i in DenseNet_169 MobileNetV2 VGG_16
	do
		python classification_test.py Plane $i 
done