#!/bin/bash

for file in *.csv; do
    if [ -s "$file" ]; then
        echo "Keeping $file"
    else
        echo "Deleting empty file: $file"
        rm "$file"
    fi
done