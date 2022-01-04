import os

path = '/home/zq/work/data/underground_mine/meibi/txt_train/'
files = os.listdir(path)
sum_0, sum1, sum2, sum3, sum4, sum5 = 0, 0, 0, 0, 0, 0
l = []
for file in files:
    with open(path + file, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            l.append(line.split()[0])

sum_helmet_yes = l.count('0')
sum_helmet_no = l.count('1')
sum_person = l.count('2')
sum_coculater = l.count('3')
sum_coculater_roller = l.count('4')
sum_foreign_matter = l.count('5')

print("sum_helmet_yes: {}".format(sum_helmet_yes))
print("sum_helmet_no: {}".format(sum_helmet_no))
print("sum_person: {}".format(sum_person))
print("sum_coculater: {}".format(sum_coculater))
print("sum_coculater_roller: {}".format(sum_coculater_roller))
print("sum_foreign_matter: {}".format(sum_foreign_matter))
