from ctypes import sizeof
import numpy
import string
import sys

from pandas import array

unwantedChars = ['[', ']', '(', ')', '=', '_', ':']
wantedChars = ['e','E']

filename = sys.argv[1]


lineCount = 0
num_lines = sum(1 for line in open(filename))
arraySize = 0

f = open(filename, "r")
for x in f:
    if lineCount == 1:
        x = x[8:]
    if lineCount == num_lines-1:
        x = x[:-22]
    if lineCount >= 1 and lineCount < num_lines:
        listWords = x.split(",")
        for word in listWords:
            for character in string.ascii_lowercase:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            for character in string.ascii_uppercase:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            for character in unwantedChars:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            word = word.strip()

            if (word != ''):
                print("|" + word + "|"  + (str)(float(word)))
                arraySize += 1
    lineCount += 1


outfile = open(filename.replace(".txt",".h"), "w") 
header = "extern const float in_1[z] = { \\\n"
print(arraySize)
header = header.replace("z",str(arraySize))
outfile.write(header)

print("------------------------------------------------------------------")

lineCount = 0
num_lines = sum(1 for line in open(filename))
currentSize = 0

f = open(filename, "r")
for x in f:
    if lineCount == 1:
        x = x[8:]
    if lineCount == num_lines-1:
        x = x[:-22]
    if lineCount >= 1 and lineCount < num_lines:
        listWords = x.split(",")
        for word in listWords:
            for character in string.ascii_lowercase:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            for character in string.ascii_uppercase:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            for character in unwantedChars:
                if (wantedChars.__contains__(character) == False):
                    word = word.replace(character, '')
            word = word.strip()
            if (word != ''):
                if currentSize == arraySize-1:
                    outfile.write(str(float(word)))
                    outfile.write(" };")
                else:
                     outfile.write(str(float(word)))
                     outfile.write(", \ \n")
                currentSize += 1
    lineCount += 1

outfile.close()

            

