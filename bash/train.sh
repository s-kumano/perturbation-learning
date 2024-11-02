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
  f=${data_root}/${mode}/${f1}_${f2}

  if [ $mode == synthetic ]; then
    subargs="synthetic $in_dim $data_gen_method $n_sample"
  else
    subargs="real $dataset_name"
  fi

  echo $f >> logs/${now}.out 2>&1

  if [ -e $f ]; then
    if [ $for_cossim ]; then
      f=${f}/advs
      run=$([[ -e $f ]] && echo false || echo true)
    elif [ $for_map ]; then
      f=${f}/classifier
      run=$([[ -e $f ]] && echo false || echo true)
    else
      run=false
    fi
  else
    run=true
  fi

  if $run; then
    python3 train.py \
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
      $on_original \
      $for_cossim \
      $for_map \
      $subargs \
      >> logs/${now}.out 2>&1
  fi
}

################### Synthetic
mode=synthetic

################### Shifted Gauss (a)
in_dim=100
hidden_dim=100
slope=0.0
data_gen_method=shifted_gauss
n_sample=1000
loss_name=identity
epochs_1=1000
epochs_2=1000
perturbation_size=0.01 # sqrt(d*.001**2)
seed=0
on_original=""
for_map=""

for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do

for_cossim=-c
in_dim_and_perturbation_size_list=("10 0.0031" "20 0.0044" "30 0.0054" "40 0.0063" "50 0.007" "60 0.0077" "70 0.0083" "80 0.0089" "90 0.0094" "100 0.01")
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
in_dim=100
perturbation_size=0.01
for_cossim=""

loss_name=logistic
in_dim_and_perturbation_size_list=("10 0.0031" "15 0.0038" "20 0.0044" "25 0.005" "30 0.0054" "35 0.0059" "40 0.0063" "45 0.0067" "50 0.007")
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
in_dim=100
loss_name=identity
perturbation_size=0.01

hidden_dims=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

n_samples=(2 4 6 8 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for n_sample in "${n_samples[@]}"; do main; done
n_sample=1000

epochs_2s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=1000

perturbation_sizes=(0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=0.01

done
done

for_cossim=-c
lr_1=0.1
epochs_1s=(1 2 3 4 5 6 7 8 9 10)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
for_cossim=""

in_dim=100
hidden_dim=100
slope=0.0
data_gen_method=shifted_gauss
n_sample=1000
loss_name=identity
lr_1=1.0
lr_2=1.0
epochs_1=100
epochs_2=100
perturbation_size=0.01 # sqrt(d*.001**2)
seed=0
on_original=""
for_map=-m
main

################### Shifted Gauss (b)
in_dim=5000
hidden_dim=100
slope=0.0
data_gen_method=shifted_gauss
n_sample=10000
loss_name=identity
epochs_1=1000
epochs_2=1000
perturbation_size=0.7 # sqrt(d*.01**2)
seed=1
# seed=0 produces unnatural high agreements between the network predictions
# even in a non-perturbation environment.
# As far as I can see, this is not a bug, just luck.
on_original=-o
for_map=""

for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do

in_dim_and_perturbation_size_list=("10 0.031" "20 0.044" "30 0.054" "40 0.063" "50 0.07" "60 0.077" "70 0.083" "80 0.089" "90 0.094" "100 0.1" "200 0.14" "300 0.17" "400 0.2" "500 0.22" "600 0.24" "700 0.26" "800 0.28" "900 0.3" "1000 0.31" "2000 0.44" "3000 0.54" "4000 0.63" "5000 0.7")
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
in_dim=5000
perturbation_size=0.7

hidden_dims=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

n_samples=(2 4 6 8 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for n_sample in "${n_samples[@]}"; do main; done
n_sample=10000

perturbation_sizes=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=0.7

done
done

lr_1=0.01
epochs_1s=(1 2 3 4 5 6 7 8 9 10)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
epochs_1=1000

lr_2=0.01
epochs_2s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_1 in 1.0 0.1; do
for epochs_2 in "${epochs_2s[@]}"; do
  main
done
done
epochs_2=1000

loss_name=logistic
in_dim_and_perturbation_size_list=("10 0.031" "20 0.044" "30 0.054" "40 0.063" "50 0.07" "60 0.077" "70 0.083" "80 0.089" "90 0.094" "100 0.1" "200 0.14" "300 0.17" "400 0.2" "500 0.22" "600 0.24" "700 0.26" "800 0.28" "900 0.3" "1000 0.31" "2000 0.44" "3000 0.54" "4000 0.63" "5000 0.7")
for lr_1 in 10.0 5.0 1.0 0.1 0.01; do
for lr_2 in 10.0 5.0 1.0 0.1 0.01; do
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
done
done

in_dim=100
hidden_dim=100
slope=0.0
data_gen_method=shifted_gauss
n_sample=5000
loss_name=identity
lr_1=1.0
lr_2=1.0
epochs_1=100
epochs_2=100
perturbation_size=0.1 # sqrt(d*.01**2)
seed=1
on_original=-o
for_map=-m
main

################### Gauss (a)
in_dim=10000
hidden_dim=100
slope=0.0
data_gen_method=gauss
n_sample=10000
loss_name=identity
epochs_1=1000
epochs_2=1000
perturbation_size=0.1 # sqrt(d*.001**2)
seed=0
on_original=""
for_map=""

for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do

for_cossim=-c
in_dim_and_perturbation_size_list=("100 0.01" "200 0.014" "300 0.017" "400 0.02" "500 0.022" "600 0.024" "700 0.026" "800 0.028" "900 0.03" "1000 0.031" "2000 0.044" "3000 0.054" "4000 0.063" "5000 0.07" "6000 0.077" "7000 0.083" "8000 0.089" "9000 0.094" "10000 0.1")
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
in_dim=10000
perturbation_size=0.1
for_cossim=""

loss_name=logistic
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
in_dim=10000
loss_name=identity
perturbation_size=0.1

hidden_dims=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

n_samples=(2 4 6 8 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for n_sample in "${n_samples[@]}"; do main; done
n_sample=10000

epochs_2s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=1000

perturbation_sizes=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=0.1

done
done

for_cossim=-c
lr_1=0.1
epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
for_cossim=""

in_dim=1000
hidden_dim=1000
slope=0.0
data_gen_method=gauss
n_sample=2000
loss_name=identity
lr_1=1.0
lr_2=1.0
epochs_1=1000
epochs_2=1000
perturbation_size=0.031 # sqrt(d*.001**2)
seed=0
on_original=""
for_map=-m
main

################### Gauss (b)
in_dim=10000
hidden_dim=100
slope=0.0
data_gen_method=gauss
n_sample=10000
loss_name=identity
epochs_1=1000
epochs_2=1000
perturbation_size=10.0 # sqrt(d*.1**2)
seed=0
on_original=-o
for_map=""

for lr_1 in 1.0 0.1; do
for lr_2 in 1.0 0.1; do

in_dim_and_perturbation_size_list=("100 1.0" "200 1.4" "300 1.7" "400 2.0" "500 2.2" "600 2.4" "700 2.6" "800 2.8" "900 3.0" "1000 3.1" "2000 4.4" "3000 5.4" "4000 6.3" "5000 7.0" "6000 7.7" "7000 8.3" "8000 8.9" "9000 9.4" "10000 10.0")
for in_dim_and_perturbation_size in "${in_dim_and_perturbation_size_list[@]}"; do
for loss_name in identity logistic; do
  s=($in_dim_and_perturbation_size)
  in_dim=${s[0]}
  perturbation_size=${s[1]}
  main
done
done
in_dim=10000
loss_name=identity
perturbation_size=10.0

hidden_dims=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

n_samples=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for n_sample in "${n_samples[@]}"; do main; done
n_sample=10000

perturbation_sizes=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=10.0

done
done

lr_1=0.1
epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_2 in 1.0 0.1; do
for epochs_1 in "${epochs_1s[@]}"; do
  main
done
done
epochs_1=1000

lr_2=0.1
epochs_2s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for lr_1 in 1.0 0.1; do
for epochs_2 in "${epochs_2s[@]}"; do
  main
done
done
epochs_2=1000

in_dim=1000
hidden_dim=1000
slope=0.0
data_gen_method=gauss
n_sample=10000
loss_name=identity
lr_1=0.1
lr_2=0.1
epochs_1=1000
epochs_2=1000
perturbation_size=0.31 # sqrt(d*.01**2)
seed=0
on_original=-o
for_map=-m
main

################### Real
mode=real

################### MNIST (a)
dataset_name=MNIST
hidden_dim=100
slope=0.0
loss_name=identity
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=0
on_original=""
for_map=""

for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do

hidden_dims=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

for_cossim=-c
epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_1 in "${epochs_1s[@]}"; do main; done
epochs_1=100
for_cossim=""

epochs_2s=(10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=100

perturbation_sizes=(0.028 0.042 0.056 0.07 0.084 0.098 0.112 0.126 0.14)
for loss_name in identity logistic; do
for perturbation_size in "${perturbation_sizes[@]}"; do
  main
done
done
loss_name=identity
perturbation_size=0.14

done
done

dataset_name=MNIST
hidden_dim=1000
slope=0.0
loss_name=identity
lr_1=0.01
lr_2=0.01
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=0
on_original=""
for_map=-m
main

################### MNIST (b)
dataset_name=MNIST
hidden_dim=100
slope=0.0
loss_name=identity
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=1 # see above
on_original=-o
for_map=""

for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do

hidden_dims=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_1 in "${epochs_1s[@]}"; do main; done
epochs_1=100

epochs_2s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=100

perturbation_sizes=(0.0 0.014 0.028 0.042 0.056 0.07 0.084 0.098 0.112 0.126 0.14)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=0.14

loss_name=logistic
seed=2 # see above
perturbation_sizes=(0.0 0.14 0.28 0.42 0.56 0.7 0.84 0.98 1.12 1.26 1.4)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
loss_name=identity
perturbation_size=0.14
seed=1

done
done

dataset_name=MNIST
hidden_dim=1000
slope=0.0
loss_name=identity
lr_1=0.01
lr_2=0.01
epochs_1=1000
epochs_2=1000
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=1
on_original=-o
for_map=-m
main

################### FMNIST (a)
dataset_name=FMNIST
hidden_dim=100
slope=0.0
loss_name=identity
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=0
on_original=""
for_map=""

for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do

hidden_dims=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

for_cossim=-c
epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_1 in "${epochs_1s[@]}"; do main; done
epochs_1=100
for_cossim=""

epochs_2s=(10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=100

perturbation_sizes=(0.0 0.014 0.028 0.042 0.056 0.07 0.084 0.098 0.112 0.126 0.14)
for loss_name in identity logistic; do
for perturbation_size in "${perturbation_sizes[@]}"; do
  main
done
done
loss_name=identity
perturbation_size=0.14

done
done

dataset_name=FMNIST
hidden_dim=1000
slope=0.0
loss_name=identity
lr_1=0.01
lr_2=0.01
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=0
on_original=""
for_map=-m
main

################### FMNIST (b)
dataset_name=FMNIST
hidden_dim=100
slope=0.0
loss_name=identity
epochs_1=100
epochs_2=100
perturbation_size=0.14 # sqrt(d*.01**2) / 2
seed=2 # see above
on_original=-o
for_map=""

for lr_1 in 0.01 0.001; do
for lr_2 in 0.01 0.001; do

hidden_dims=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
for hidden_dim in "${hidden_dims[@]}"; do main; done
hidden_dim=100

slopes=(0.0 0.25 0.5 0.75 0.99)
for slope in "${slopes[@]}"; do main; done
slope=0.0

epochs_1s=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)
for epochs_1 in "${epochs_1s[@]}"; do main; done
epochs_1=100

epochs_2s=(10 20 30 40 50 60 70 80 90 100)
for epochs_2 in "${epochs_2s[@]}"; do main; done
epochs_2=100

perturbation_sizes=(0.0 0.014 0.028 0.042 0.056 0.07 0.084 0.098 0.112 0.126 0.14)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
perturbation_size=0.14

loss_name=logistic
perturbation_sizes=(0.0 0.14 0.28 0.42 0.56 0.7 0.84 0.98 1.12 1.26 1.4)
for perturbation_size in "${perturbation_sizes[@]}"; do main; done
loss_name=identity
perturbation_size=0.14

done
done

dataset_name=FMNIST
hidden_dim=1000
slope=0.0
loss_name=identity
lr_1=0.001
lr_2=0.001
epochs_1=100
epochs_2=100
perturbation_size=1.4 # sqrt(d*.1**2) / 2
seed=2
on_original=-o
for_map=-m
main