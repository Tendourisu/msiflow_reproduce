#!/bin/sh

# run msi_if_registration_flow to register IFM and MSI
snakemake --snakefile msi_if_registration_flow/Snakefile --configfile demo_full/data/msi_if_registration/config.yaml --config data='demo_full/data/msi_if_registration' --cores all

# copy the registered IFM image to demo/data/if_segmentation
cp demo_full/data/msi_if_registration/registered/UPEC_12.tif demo_full/data/if_segmentation

# run if_segmentation_flow to segment the Ly6G image channel
snakemake --snakefile if_segmentation_flow/Snakefile --configfile demo_full/data/if_segmentation/config.yaml --config data='demo_full/data/if_segmentation' --cores all

# copy the registered and segmented image to demo/data/ly6g_molecular_signatures/bin_imgs
mkdir demo_full/data/ly6g_molecular_signatures/bin_imgs
cp demo_full/data/if_segmentation/segmented/UPEC_12.tif demo_full/data/ly6g_molecular_signatures/bin_imgs
# run molecular_signatures_flow to extract lipidomic signatures of Ly6G (neutrophils)
snakemake --snakefile molecular_signatures_flow/Snakefile --configfile demo_full/data/ly6g_molecular_signatures/config.yaml --config data='demo_full/data/ly6g_molecular_signatures' --cores all


snakemake --snakefile region_group_analysis_flow/Snakefile --configfile demo_full/data/region_group_analysis/config.yaml --config data='demo_full/data/region_group_analysis' --cores all