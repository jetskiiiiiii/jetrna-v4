#!/bin/bash

RNASTRALIGN_DEST_DIR="./dataset/ct/rnastralign"
ARCHIVEII_DEST_DIR="./dataset/ct/archiveII"

RNASTRALIGN_SOURCE_DIR="/Volumes/VILLAIN/CAPSTONE/DATASETS/RNAStralign"
ARCHIVEII_SOURCE_DIR="/Volumes/VILLAIN/CAPSTONE/DATASETS/ArchiveII"

find $RNASTRALIGN_SOURCE_DIR -type f -name "*.ct" -exec mv -v {} "$RNASTRALIGN_DEST_DIR" \;
find $ARCHIVEII_SOURCE_DIR -type f -name "*.ct" -exec mv -v {} "$ARCHIVEII_DEST_DIR" \;
