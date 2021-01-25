#!/bin/sh

zip -F weights/weights.zip --out weights/weights_unsplit.zip
unzip weights/weights_unsplit.zip
rm weights/weights_unsplit.zip