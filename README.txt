




1) create yaml file from environment:
# documentation: 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
```
conda activate myenv
conda env export > environment.yml
```


2) create conda environment from yaml file:
# documentation: 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
```
conda env create -f environment.yml
```

veryfiy that environment was created:
``` 
conda list
```







useful links:
NEC implementation in tensorflow:
    https://github.com/imai-laboratory/nec
    
    

	