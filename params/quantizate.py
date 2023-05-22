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

if len(sys.argv) > 2:
    n_bins = int(sys.argv[1])
    mode = sys.argv[2]
else:
    print("USAGE python quantizate.py NUM_BINS MODE MULTIPLICATOR")
    print("e.g. python quantizate.py 16 fixed MULTIPLICATOR")
    quit(1)


## Load all data
data_0_weight = extract_floats("0_weight.h")
data_3_weight = extract_floats("3_weight.h")
data_6_weight = extract_floats("6_weight.h")
data_fc1_weight = extract_floats("classifier_1_weight.h")
data_fc2_weight = extract_floats("classifier_2_weight.h")

data_0_bias = extract_floats("0_bias.h")
data_3_bias = extract_floats("3_bias.h")
data_6_bias = extract_floats("6_bias.h")
data_fc1_bias = extract_floats("classifier_1_bias.h")
data_fc2_bias = extract_floats("classifier_2_bias.h")

data = np.concatenate([data_0_weight, data_3_weight ])
data = np.concatenate([data, data_6_weight ])
data = np.concatenate([data, data_fc1_weight ])
data = np.concatenate([data, data_fc2_weight ])
data = np.concatenate([data, data_0_bias ])
data = np.concatenate([data, data_3_bias ])
data = np.concatenate([data, data_6_bias ])
data = np.concatenate([data, data_fc1_bias ])
data = np.concatenate([data, data_fc2_bias ])


########## LUT
if mode == "fixed":
    print("MODE : FIXED")
    print("BINS : "+ str(n_bins))
    bin_ranges, lookup_table = calculate_bin_ranges(data, n_bins)
if mode == "variable":
    print("MODE : VARIABLE")
    print("BINS : "+ str(n_bins))
    bin_ranges, lookup_table = calculate_variable_bins(data, n_bins)


# salva os indices em um arquivo C
indices_filename = "bins.txt"
with open(indices_filename, "w") as f:
    for start, end in bin_ranges:
        f.write(f"Bin: [{start:.4f}, {end:.4f}]\n")

lookup_table_str = "const float params_lut[%d] = {" % len(lookup_table) + "\n".join([f"{val:.4f}," for val in lookup_table])[:-1] + "};\n"
# lookup_table_str = "const int params_lut[%d] = {" % len(lookup_table) + "\n".join([f"{val:.4f}," for val in lookup_table])[:-1] + "};\n"


# salva a tabela de busca em um arquivo C
lookup_table_filename = "model_params_lut.h"
with open(lookup_table_filename, "w") as f:
    f.write(lookup_table_str)
print(f"Tabela de busca salva em {lookup_table_filename}")



########## Indices


arrayName = "conv0_bias"
indices = bin_indices(data_0_bias, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "0_bias_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "conv0_weights"
indices = bin_indices(data_0_weight, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "0_weight_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "conv3_bias"
indices = bin_indices(data_3_bias, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "3_bias_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "conv3_weights"
indices = bin_indices(data_3_weight, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "3_weight_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "conv6_bias"
indices = bin_indices(data_6_bias, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "6_bias_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "conv6_weights"
indices = bin_indices(data_6_weight, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "6_weight_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "fc1_bias"
indices = bin_indices(data_fc1_bias, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "classifier_1_bias_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "fc1_weights"
indices = bin_indices(data_fc1_weight, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "classifier_1_weight_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "fc2_bias"
indices = bin_indices(data_fc2_bias, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "classifier_2_bias_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")




arrayName = "fc2_weights"
indices = bin_indices(data_fc2_weight, bin_ranges)
indices_str = "const unsigned char " + arrayName + "_indices[%d] = {" % len(indices) + "\n".join([f"{val}," for val in indices])[:-1] + "};\n"

# salva os indices em um arquivo C
indices_filename = "classifier_2_weight_indices.h"
with open(indices_filename, "w") as f:
    f.write(indices_str)
print(f"Indices salvos em {indices_filename}")








