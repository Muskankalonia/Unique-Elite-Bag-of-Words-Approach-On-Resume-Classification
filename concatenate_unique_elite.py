import pandas as pd
import os
def create_dictionary(csv_paths, x):
    token = {}
    for  i, csv in enumerate(csv_paths):
        print(i, x[i])
        j = x[i]
        df = pd.read_csv(csv)
        filtered_df = df[df['entropy']!="[0.0]"]
        selected_entries = filtered_df.head(j)
        print(len(selected_entries))
        token.update({word: idx for idx, word in enumerate(selected_entries['words'])})
    return token

csv_paths = ['unique elite dataset 2/Database_Administrator_entropy.csv']

x = [3616]
    
# x = 'Unique_elite_datset1/'
# csv_paths = [ x+ 'accountant_entropy.csv',x+'advocate_entropy.csv',x+'agriculture_entropy.csv',x+'apparel_entropy.csv',x+'art_entropy.csv'
#              ,x+'automobile_entropy.csv',x+'aviation_entropy.csv',x+'banking_entropy.csv',x+'bpo_entropy.csv',x+'business_entropy.csv'
#              ,x+'chef_entropy.csv',x+'construction_entropy.csv',x+'consultant_entropy.csv',x+'designer_entropy.csv',x+'digital_entropy.csv'
#              ,x+'engineering_entropy.csv',x+'finance_entropy.csv',x+'fitness_entropy.csv',x+'healthcare_entropy.csv',x+'hr_entropy.csv'
#              ,x+'info_entropy.csv',x+'public_entropy.csv',x+'sales_entropy.csv',x+'teacher_entropy.csv']

# y = [1157, 1291, 1054, 1093, 1130, 907, 1419, 1265, 694, 1182, 1338, 1245, 1378, 1193, 1021, 1407, 1114, 1203, 1113, 1025, 1478, 1301, 1127, 1036]

# for i in range(len(csv_paths)):
#     print(csv_paths[i], y[i])
token = create_dictionary(csv_paths,x)
print(token)
import json
my_json = json.dumps(token)
f = open("unique elite dataset 2/database_administrator_unique_elite.json","w")
f.write(my_json)
f.close()