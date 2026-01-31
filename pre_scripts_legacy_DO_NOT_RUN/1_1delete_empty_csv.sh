#!/bin/bash

for file in $(find summer23/raw/ -type f -name "*.csv"); do
    if [ -s "$file" ]; then
        echo "Keeping $file"
    else
        echo "Deleting empty file: $file"
        rm "$file"
    fi
done