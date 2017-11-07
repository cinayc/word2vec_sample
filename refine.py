import re

def refine(filename):
    space_pattern = re.compile(r'[,!"#\$%&()*+,-./:;<=>?@\[\]\^_`{|}~\r\t\n‘’“”·•…ㆍ㈜▲↑←→ㄴ\'°′◆″]+', re.DOTALL)
    fold_pattern = re.compile(r'[ ]{2,}', re.DOTALL)
    with open(filename, 'r') as fr:
        input = ''
        with open(filename + '_refine', 'w') as fw:
            while True:
                line = fr.readline()
                if not line:
                    break

                line = line.strip()
                if len(line) > 0:
                    # print('o: %s' % line)
                    line = re.sub(space_pattern, ' ', line, 0)
                    # print('s: %s' % line)
                    line = re.sub(fold_pattern, ' ', line, 0)
                    print('f: %s' % line)
                    fw.write(line+'\n')
                else:
                    # print('b: %s' % line)
                    pass




# filename = "alice.txt"
# filename = "data_sample"
filename = "/data/forW2V/data.txt"
refine(filename)
