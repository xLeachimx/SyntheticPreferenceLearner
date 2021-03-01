import os

def main():
    with open('conv_mistakes.dat','w') as fout:
        fout.write("Mistakes\n")
    for fname in os.listdir('./'):
        ext = fname.split('.')[-1]
        if ext != "txt":
            continue
        count = {}
        with open(fname,'r') as fin:
            for line in fin:
                line = line.strip()
                if line != '':
                    if line not in count:
                        count[line] = 1
                    else:
                        count[line] += 1
        with open('conv_mistakes.dat','a') as fout:
            fout.write("START\n")
            for key in count.keys():
                fout.write(key+':'+str(count[key])+"\n")
            fout.write("END\n")

main()
