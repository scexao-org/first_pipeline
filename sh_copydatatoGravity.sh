#!/bin/bash

echo "Copying all data to remote gravity server:"
rsync -avz --progress --include='*/' --include='preproc/***' --exclude='*' --prune-empty-dirs $DETDATA/. gravity_sl:/volumes/dio/LANTERNE/.
