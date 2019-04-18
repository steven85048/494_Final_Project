import csv

reader = csv.reader(open('../data/suicide.csv'))
csv_lines = list(reader)

num_rows = len( csv_lines )
for i in range( 1, num_rows ):
    comma_remove = csv_lines[i][9].replace(',', '')
    csv_lines[i][9] = int(comma_remove)

writer = csv.writer(open('../data/suicide_cleaned.csv', 'w+'))
writer.writerows(csv_lines)