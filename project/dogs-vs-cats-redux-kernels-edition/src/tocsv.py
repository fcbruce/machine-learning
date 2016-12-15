#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Thu 15 Dec 2016 19:21:15
#
#

print 'id,label'

for i in range(12500):
    line = raw_input()
    if line.find('True') >= 0:
        print "%d,%d" % (i + 1, 0)
    else:
        print "%d,%d" % (i + 1, 1)

