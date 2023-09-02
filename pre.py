import re

file = open("spa-eng/spa.txt", "r")

lines = file.readlines()

file2 = open("spanish_million.txt", "w")

for line in lines:
    words = re.split(r'\t+', line)
    file2.write(words[1] + "\n")
