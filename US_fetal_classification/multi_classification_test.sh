#!/bin/sh
for i in DenseNet_169 MobileNetV2
	do
		python classification_test.py Plane $i 
done