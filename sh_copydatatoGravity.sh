#!/bin/bash

echo "Copying all data to remote gravity server:"
rsync -avz --progress --include='*/' --include='wavemaps/***' --exclude='*' --prune-empty-dirs $DETDATA/. gravity_sl:/volumes/dio/LANTERNE/.
rsync -avz --progress --include='*/' --include='flatmaps/***' --exclude='*' --prune-empty-dirs $DETDATA/. gravity_sl:/volumes/dio/LANTERNE/.
rsync -avz --progress --include='*/' --include='pixelmaps/***' --exclude='*' --prune-empty-dirs $DETDATA/. gravity_sl:/volumes/dio/LANTERNE/.
rsync -avz --progress --include='*/' --include='preproc/***' --exclude='*' --prune-empty-dirs $DETDATA/. gravity_sl:/volumes/dio/LANTERNE/.
