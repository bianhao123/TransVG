s = input('输出要转换的字符串')

s_list = s.split(' ')
for i in s_list:
    print(f'"{i}"', end=', ')
    # print('\"', i, '\"', ',', end='')
# print(s_list)
