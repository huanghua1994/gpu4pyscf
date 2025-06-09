import sys

def parse_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    atom_count = int(lines[0].strip())
    atom_lines = lines[2:2+atom_count]

    element_counts = {}
    for line in atom_lines:
        parts = line.split()
        element = parts[0]
        element_counts[element] = element_counts.get(element, 0) + 1

    formula = ''
    if 'C' in element_counts:
        formula += 'C' + (str(element_counts['C']) if element_counts['C'] > 1 else '')
        element_counts.pop('C')
    if 'H' in element_counts:
        formula += 'H' + (str(element_counts['H']) if element_counts['H'] > 1 else '')
        element_counts.pop('H')

    for element in sorted(element_counts.keys()):
        count = element_counts[element]
        formula += element + (str(count) if count > 1 else '')

    return formula

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <xyz-file>")
        sys.exit(1)

    filename = sys.argv[1]
    formula = parse_xyz(filename)
    print('', formula)

