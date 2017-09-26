import re

def read_input(filename):
    remove_pattern = re.compile(r'[,!"#\$%&()*+,-./:;<=>?@\[\]\^_`{|}~\r\t\n‘’“”]+', re.DOTALL)
    space_patten = re.compile(r'[,"#\$%&*+,-./:;<=>?@\^_`|~]{2,}', re.DOTALL)
    with open(filename, 'r') as f:
        input = ''
        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip()
            if len(line) > 0:
                print('o: %s' % line)
                line = re.sub(space_patten, ' ', line, 0)
                print('s: %s' % line)
                line = re.sub(remove_pattern, '', line, 0)
                print('r: %s' % line)
                input += line + ' '
            else:
                # print('b: %s' % line)
                pass
        return input

def write_input(filename, input):
    with open(filename+'_refine', 'w') as f:
        f.write(input)

filename = "alice.txt"

input = read_input(filename)
write_input(filename, input)
