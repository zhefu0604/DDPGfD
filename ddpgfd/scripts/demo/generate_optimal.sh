#!/bin/bash

OUTFILE=("2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_0_4597"
         "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_1_4927"
         "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825"
         "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_1_4938"
         "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_0_4523"
         "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_1_4582"
         "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_0_5672"
         "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_1_4817"
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

for index in "${!OUTFILE[@]}"; do
  python generate_samples.py "optimal/emissions/${OUTFILE[$index]}.csv" "optimal/${OUTFILE[$index]}.pkl"
done