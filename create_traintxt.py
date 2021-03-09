from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('Project/Chess/images') if isfile(join('Project/Chess/images', f))]

newlist = [string for string in onlyfiles if string[-3:]!='txt']

print(newlist)

with open('traintxt.txt', 'w') as f:
    for item in newlist:
        f.write('Project/Chess/images/'+ item + "\n")