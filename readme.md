This is the official code for "Wide Two-Layer Networks can Learn from Adversarial Perturbations" [S. Kumano et al., NeurIPS24].

All generated data, including synthetic datasets, model parameters, and adversarial examples, can be downloaded from [here](https://drive.google.com/file/d/10zM3Rc2uV9eHqGKowDdEbk3Cv_wl0pJU/view?usp=drive_link) or [here](https://filedn.com/lAlreeY65CBjFVbAkaD5F7k/Research/%5BNeurIPS24%5D%20Wide%20Two-Layer%20Networks%20can%20Learn%20from%20Adversarial%20Perturbations/data.zip) (48GB).

# Setup
```console
docker-compose -f "docker/docker-compose.yaml" up -d --build 
```

# Run
```console
bash/train.sh <gpu_id: int>
bash/cossim.sh <gpu_id: int>
bash/map.sh <gpu_id: int>
```