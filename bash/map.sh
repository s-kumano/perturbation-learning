#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){

  on_original_txt=$([[ $on_original ]] && echo True || echo False)

  if [ $mode == synthetic ]; then
    f1=${in_dim}_${hidden_dim}_${slope}_${data_gen_method}_${n_sample}_${loss_name}_${lr_1}_${lr_2}
    f2=${epochs_1}_${epochs_2}_${perturbation_size}_${seed}_${on_original_txt}
  else
    f1=${dataset_name}_${hidden_dim}_${slope}_${loss_name}_${lr_1}_${lr_2}
    f2=${epochs_1}_${epochs_2}_${perturbation_size}_${seed}_${on_original_txt}
  fi
  f=${data_root}/${mode}/${f1}_${f2}/map

  if [ $mode == synthetic ]; then
    subargs="synthetic $in_dim $data_gen_method $n_sample"
  else
    subargs="real $dataset_name"
  fi

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 map.py \
      $hidden_dim \
      $slope \
      $loss_name \
      $lr_1 \
      $lr_2 \
      $epochs_1 \
      $epochs_2 \
      $perturbation_size \
      $seed \
      $limits \
      $device \
      $on_original \
      $subargs \
      >> logs/${now}.out 2>&1
  fi
}

slope=0.0
loss_name=identity

################### Synthetic
mode=synthetic

################### Shifted Gauss (a)
in_dim=100
hidden_dim=100
data_gen_method=shifted_gauss
n_sample=1000
lr_1=1.0
lr_2=1.0
epochs_1=100
epochs_2=100
perturbation_size=0.01
seed=0
limits="-7 7"
on_original=""
main

################### Shifted Gauss (b)
in_dim=100
hidden_dim=100
data_gen_method=shifted_gauss
n_sample=5000
lr_1=1.0
lr_2=1.0
epochs_1=100
epochs_2=100
perturbation_size=0.1
seed=1
limits="-8 8"
on_original=-o
main

################### Gauss (a)
in_dim=1000
hidden_dim=1000
data_gen_method=gauss
n_sample=2000
lr_1=1.0
lr_2=1.0
epochs_1=1000
epochs_2=1000
perturbation_size=0.031
seed=0
limits="-4 5"
on_original=""
main

################### Gauss (b)
in_dim=1000
hidden_dim=1000
data_gen_method=gauss
n_sample=10000
lr_1=0.1
lr_2=0.1
epochs_1=1000
epochs_2=1000
perturbation_size=0.31
seed=0
limits="-4 5"
on_original=-o
main

################### Real
mode=real

################### MNIST (a)
dataset_name=MNIST
hidden_dim=1000
lr_1=0.01
lr_2=0.01
epochs_1=100
epochs_2=100
perturbation_size=0.14
seed=0
limits="0 10"
on_original=""
main

################### MNIST (b)
dataset_name=MNIST
hidden_dim=1000
lr_1=0.01
lr_2=0.01
epochs_1=1000
epochs_2=1000
perturbation_size=0.14
seed=1
limits="0 10"
on_original=-o
main

################### FMNIST (a)
dataset_name=FMNIST
hidden_dim=1000
lr_1=0.01
lr_2=0.01
epochs_1=100
epochs_2=100
perturbation_size=0.14
seed=0
limits="-2 13"
on_original=""
main

################### FMNIST (b)
dataset_name=FMNIST
hidden_dim=1000
lr_1=0.001
lr_2=0.001
epochs_1=100
epochs_2=100
perturbation_size=1.4
seed=2
limits="-2 13"
on_original=-o
main