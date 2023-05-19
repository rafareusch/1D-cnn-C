import sys

def multiply_values(file_path, output_file_path,multip):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("extern const float") == False:
            values_start = 1
            values_end = line.rfind("}")
            values = line[values_start:values_end].replace("\\", "").split(",")
            multiplied_values = []
            for value in values:
                value = value.strip()
                if value:
                    if line.__contains__("-"):
                        negativeMult = -1
                    else:
                        negativeMult = 1
                    multiplied_values.append(str(int(float(value) * float(multip) * float(negativeMult))))
            if (line.__contains__(";")):
                new_line = " \\\n".join(multiplied_values) + "};" 
            else:
                new_line = " \\\n".join(multiplied_values) + ",\\\n" 
            new_lines.append(new_line)
        else:
            new_lines.append(line.replace("float","int"))

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(new_lines)



# Verifica se o número correto de argumentos foi passado
if len(sys.argv) != 3:
    print("Uso: python script.py <input_file> <multip_value>")
    sys.exit(1)

input_file = sys.argv[1]
multip = sys.argv[2]
output_file = input_file.replace(".h","_int.h")
multiply_values(input_file, output_file,multip)
