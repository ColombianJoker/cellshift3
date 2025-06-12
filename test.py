from cellshift import get_file_size as getsize
from cellshift import destroy
from os.path import isfile

big_file="Casos_sin_limpiar.csv"
print(f"Tamaño de '{big_file}': {getsize(big_file)/(1024*1024):.1f} MB")
if destroy(big_file, verbose=True):
    if isfile(big_file):
        print(f"Tamaño de '{big_file}': {getsize(big_file)/(1024*1024):.1f} MB")
    else:
        print(f"No se encontró '{big_file}'!")
else:
    print(f"No se pudo destruir '{big_file}'")