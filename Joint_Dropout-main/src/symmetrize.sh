#!/bin/bash

# Paths to inputs and outputs
FORWARD_FILE="data/output/akk-en.fwd"
REVERSE_FILE="data/output/akk-en.rev"
SYMMETRIZED_FILE="data/output/akk-en.sym"

# Symmetrization
echo "Symmetrizing alignments..."
atools -i $FORWARD_FILE -j $REVERSE_FILE -c grow-diag-final-and > $SYMMETRIZED_FILE

echo "Symmetrization completed: $SYMMETRIZED_FILE"
