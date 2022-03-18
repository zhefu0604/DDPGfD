#!/bin/bash

INFILE=("1647579100_17Mar22_21h51m40s"
        "1647579120_17Mar22_21h52m00s"
        "1647579146_17Mar22_21h52m26s"
        "1647579177_17Mar22_21h52m57s"
        "1647579200_17Mar22_21h53m20s"
        "1647579221_17Mar22_21h53m41s"
        "1647579243_17Mar22_21h54m03s"
        "1647579269_17Mar22_21h54m29s"
        "1647579292_17Mar22_21h54m52s"
        "1647579315_17Mar22_21h55m15s"
        "1647579364_17Mar22_21h56m04s"
        "1647579386_17Mar22_21h56m26s"
        "1647579407_17Mar22_21h56m47s"
        "1647579427_17Mar22_21h57m07s"
        "1647579446_17Mar22_21h57m26s"
        "1647579466_17Mar22_21h57m46s"
        "1647579489_17Mar22_21h58m09s"
        "1647579508_17Mar22_21h58m28s"
        "1647579528_17Mar22_21h58m48s"
        "1647579550_17Mar22_21h59m10s"
        "1647579572_17Mar22_21h59m32s"
        "1647579593_17Mar22_21h59m53s"
        "1647579613_17Mar22_22h00m13s"
        "1647579636_17Mar22_22h00m36s"
        "1647579657_17Mar22_22h00m57s"
        "1647579677_17Mar22_22h01m17s"
        "1647579699_17Mar22_22h01m39s"
        "1647579720_17Mar22_22h02m00s"
        "1647579740_17Mar22_22h02m20s"
        "1647579761_17Mar22_22h02m41s"
        "1647579782_17Mar22_22h03m02s"
        "1647579803_17Mar22_22h03m23s"
        "1647579825_17Mar22_22h03m45s"
        "1647579874_17Mar22_22h04m34s"
        "1647579901_17Mar22_22h05m01s"
        "1647579921_17Mar22_22h05m21s"
        "1647579942_17Mar22_22h05m42s"
        "1647579964_17Mar22_22h06m04s"
        "1647579986_17Mar22_22h06m26s"
        "1647580005_17Mar22_22h06m45s"
        "1647580031_17Mar22_22h07m11s"
        "1647580052_17Mar22_22h07m32s"
        "1647580124_17Mar22_22h08m44s"
        "1647580154_17Mar22_22h09m14s"
        "1647580175_17Mar22_22h09m35s"
        "1647580198_17Mar22_22h09m58s"
        "1647580222_17Mar22_22h10m22s"
        "1647580243_17Mar22_22h10m43s"
        "1647580265_17Mar22_22h11m05s"
        "1647580285_17Mar22_22h11m25s"
        "1647580311_17Mar22_22h11m51s"
        "1647580334_17Mar22_22h12m14s"
        "1647580357_17Mar22_22h12m37s"
        "1647580378_17Mar22_22h12m58s"
        "1647580397_17Mar22_22h13m17s"
        "1647580419_17Mar22_22h13m39s")

OUTFILE=("2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_0_4597"
         "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_1_4927"
         "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825"
         "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_1_4938"
         "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_0_4523"
         "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_1_4582"
         "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_0_5672"
         "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_1_4817"
         "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917"
         "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342"
         "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_0_4463"
         "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_1_4386"
         "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_0_3977"
         "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_1_3918"
         "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_0_4223"
         "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_1_4422"
         "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_1_3778"
         "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_1_4102"
         "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_0_4937"
         "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_1_4364"
         "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_0_4540"
         "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_1_4028"
         "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_0_5016"
         "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_1_4185"
         "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_0_4200"
         "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_1_4622"
         "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_0_4125"
         "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_1_4111"
         "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_0_4357"
         "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_1_4173"
         "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_0_4427"
         "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_1_4496"
         "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294"
         "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116"
         "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_0_4101"
         "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_1_4069"
         "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_0_4796"
         "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_1_4436"
         "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_0_3889"
         "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778"
         "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_1_4387"
         "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467"
         "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483"
         "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_0_4433"
         "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_1_4288"
         "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_0_4025"
         "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_1_3973"
         "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_0_3957"
         "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_1_3621"
         "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_1_5292"
         "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_0_4595"
         "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_1_4664"
         "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_0_3836"
         "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_1_3558"
         "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_0_4190"
         "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_1_4005")

for index in "${!INFILE[@]}"; do
  python generate_samples.py "random/2-avs/accel=0.5/no-lc/emissions/${INFILE[$index]}/emissions/emissions_1.csv" "random/2-avs/accel=0.5/no-lc/${OUTFILE[$index]}.pkl"
done