import re
import sys
import numpy as np
import string


# função que extrai os floats do arquivo
def extract_floats(filename):
    with open(filename) as f:
        contents = f.read()
        floats_str = re.findall(r'-?\d+\.\d+', contents)
        floats = [float(f) for f in floats_str]
        return np.array(floats)

# função que calcula os ranges de cada bin e a tabela de busca
def calculate_bin_ranges(data, n_bins):
    data_min = np.min(data)
    data_max = np.max(data)
    bin_size = (data_max - data_min) / n_bins
    bin_ranges = []
    lookup_table = []
    for i in range(n_bins):
        bin_start = data_min + i*bin_size
        bin_end = data_min + (i+1)*bin_size
        bin_midpoint = (bin_start + bin_end) / 2
        bin_ranges.append((bin_start, bin_end))
        lookup_table.append(bin_midpoint)
    return bin_ranges, lookup_table

def calculate_variable_bins(data, n_bins):
    sorted_data = sorted(data)
    bin_sizes = np.zeros(n_bins)
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = sorted_data[0]
    for i in range(1, n_bins):
        bin_sizes[i-1] = len(sorted_data) / n_bins
        bin_edges[i] = sorted_data[int(i*bin_sizes[i-1])]
    bin_sizes[-1] = len(sorted_data) / n_bins
    bin_edges[-1] = sorted_data[-1]
    bin_ranges = []
    lookup_table = []
    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_midpoint = (bin_start + bin_end) / 2
        bin_ranges.append((bin_start, bin_end))
        lookup_table.append(bin_midpoint)
    return bin_ranges, lookup_table


# função que converte um vetor de floats em um vetor de inteiros que apontam para o bin correspondente
def bin_indices(data, bin_ranges):
    indices = np.zeros_like(data, dtype=np.int32)
    for i, (bin_start, bin_end) in enumerate(bin_ranges):
        indices[(data >= bin_start) & (data < bin_end)] = i
    return indices

# exemplo de uso

if len(sys.argv) > 4:
    filename = sys.argv[1]
    n_bins = int(sys.argv[2])
    mode = sys.argv[3]
    multiplier = int(sys.argv[4])
else:
    print("USAGE python quantizate.py FILENAME NUM_BINS MODE MULTIPLICATOR")
    print("e.g. python quantizate.py 0_weight.txt 16 fixed MULTIPLICATOR")
    quit(1)



data = extract_floats(filename)

if mode == "fixed":
    print("MODE : FIXED")
    print("BINS : "+ str(n_bins))
    bin_ranges, lookup_table = calculate_bin_ranges(data, n_bins)
if mode == "variable":
    print("MODE : VARIABLE")
    print("BINS : "+ str(n_bins))
    bin_ranges, lookup_table = calculate_variable_bins(data, n_bins)

indices = bin_indices(data, bin_ranges)



## Lógica para criar um vetor de 32 bits onde cada valor ocupa 4
N = len(indices)
tamanho_vetor_32_bits = (N + 7) // 8
indices_32bit = np.zeros(tamanho_vetor_32_bits, dtype=np.uint32)
for i in range(N):
    valor = indices[i] & 0x0F  
    indice_vetor_32_bits = i // 8
    posicao_bits = (i % 8) * 4
    indices_32bit[indice_vetor_32_bits] |= (valor << posicao_bits)





# # salva os indices em um arquivo C
# indices_filename = "bins.txt"
# with open(indices_filename, "w") as f:
#     for start, end in bin_ranges:
#         f.write(f"Bin: [{start:.4f}, {end:.4f}]\n")



if filename == "0_bias.h":
    arrayName = "conv0_bias"
if filename == "0_weight.h":
    arrayName = "conv0_weights"

if filename == "3_bias.h":
    arrayName = "conv3_bias"
if filename == "3_weight.h":
    arrayName = "conv3_weights"

if filename == "6_bias.h":
    arrayName = "conv6_bias"
if filename == "6_weight.h":
    arrayName = "conv6_weights"

if filename == "classifier_1_bias.h":
    arrayName = "fc1_bias"
if filename == "classifier_1_weight.h":
    arrayName = "fc1_weights"

if filename == "classifier_2_bias.h":
    arrayName = "fc2_bias"
if filename == "classifier_2_weight.h":
    arrayName = "fc2_weights"


lookup_table_str = "const int " + arrayName + "_lut[%d] = {" % len(lookup_table) + "\n".join([f"{val*multiplier:.0f}," for val in lookup_table])[:-1] + "};\n"
indices_str_32bit = "const unsigned int " + arrayName + "_indices_compressed[%d] = {" % len(indices_32bit) + "\n".join([f"{val}," for val in indices_32bit])[:-1] + "};\n"




# salva a tabela de busca em um arquivo C
lookup_table_filename = filename.replace(".h","_lut.h")
with open(lookup_table_filename, "w") as f:
    f.write(lookup_table_str)
print(f"Tabela de busca salva em {lookup_table_filename}")

# # salva os indices em um arquivo C
# indices_filename = filename.replace(".h","_indices.h")
# with open(indices_filename, "w") as f:
#     f.write(indices_str)
# print(f"Indices salvos em {indices_filename}")

# salva os indices em um arquivo C
indices_filename_32bit = filename.replace(".h","_indices_compressed.h")
with open(indices_filename_32bit, "w") as f:
    f.write(indices_str_32bit)
print(f"Indices salvos em {indices_filename_32bit}")
