#!/bin/sh

# run msi_preprocessing_flow to register IFM and MSI
snakemake --snakefile msi_preprocessing_flow/Snakefile --configfile demo_control/data/maldi-2-control-02/config.yaml --config data='demo_control/data/maldi-2-control-02' --cores all
