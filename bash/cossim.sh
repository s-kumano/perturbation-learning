#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){
  if [ $mode == synthetic ]; then
    f1=${in_dim}_${hidden_dim}_${slope}_${data_gen_method}_${n_sample}_${loss_name}_${lr_1}_${lr_2}
    f2=${epochs_1}_${epochs_2}_${perturbation_size}_${seed}_False
  else
    f1=${dataset_name}_${hidden_dim}_${slope}_${loss_name}_${lr_1}_${lr_2}
    f2=${epochs_1}_${epochs_2}_${perturbation_size}_${seed}_False
  fi
  f=${data_root}/${mode}/${f1}_${f2}/cossim

  if [ $mode == synthetic ]; then
    subargs="synthetic $in_dim $data_gen_method $n_sample"
  else
    subargs="real $dataset_name"
  fi

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 cossim.py \
      $hidden_dim \
      $slope \
      $loss_name \
      $lr_1 \
      $lr_2 \
      $epochs_1 \
      $epochs_2 \
      $perturbation_size \
      $seed \
      $device \
      $subargs \
      >> logs/${now}.out 2>&1
  fi
}

slope=0.0
loss_name=identity
seed=0

################### Synthetic
mode=synthetic

################### Shifted Gauss
in_dim=100
hidden_dim=100
data_gen_method=shifted_gauss
n_sample=1000
epochs_1=1000
epochs_2=1000
perturbation_size=0.01

in_dim_and_perturbation_size_list=("10 0.0031" "20 0.0044" "30 0.0054" "40 0.0063" "50 0.007" "60 0.0077" "70 0.0083" "80 0.0089" "90 0.0094" "100 0.01")
for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
done
done

lr_1=0.1
epochs_1s=(1 2 3 4 5 6 7 8 9 10)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done

################### Gauss
in_dim=10000
hidden_dim=100
data_gen_method=gauss
n_sample=10000
epochs_1=1000
epochs_2=1000
perturbation_size=0.1

in_dim_and_perturbation_size_list=("100 0.01" "200 0.014" "300 0.017" "400 0.02" "500 0.022" "600 0.024" "700 0.026" "800 0.028" "900 0.03" "1000 0.031" "2000 0.044" "3000 0.054" "4000 0.063" "5000 0.07" "6000 0.077" "7000 0.083" "8000 0.089" "9000 0.094" "10000 0.1")
for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
done
done

lr_1=0.1
epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done

################### Real
mode=real

################### MNIST
dataset_name=MNIST
hidden_dim=100
epochs_1=100
epochs_2=100
perturbation_size=0.14

epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
done

################### FMNIST
dataset_name=FMNIST
hidden_dim=100
epochs_1=100
epochs_2=100
perturbation_size=0.14

epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
done