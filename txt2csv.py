# Read the input file and convert it to csv format

import csv

def txt2csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow(line.split())

txt2csv('input.txt', 'input.csv')
