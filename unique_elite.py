import pandas as pd
import numpy as np
import math
import multiprocessing
import os

def unique_elite(acc,num_elite,category, train):
    print(num_elite, category)
    # print(acc)
    # print(train)
    token = {}
    token.update({'':0})
    counter = 0
    for i in range(0,num_elite):
        word = acc.loc[i].at["words"]
        if word not in token:
            token.update({word:counter+1})
            counter = counter+1

    result_train = np.zeros(shape = (len(train['Updated_resume']),
                            max(token.values())+1))
    for i, sample in enumerate(train['Updated_resume']):

        for j, considered_word in list(enumerate(sample.split())):

            if token.get(considered_word)!=None :
                index = token.get(considered_word)
                result_train[i, index] = result_train[i,index]+1

    x = num_elite+1
    counter = 0
    cf_arr= np.zeros(shape = (6, x))

    for i in range(0,1532):
        for j in range(0,x):
            cf_arr[0][j] = cf_arr[0][j]+result_train[i][j]

    for i in range(1533,2563):
        for j in range(0,x):
            cf_arr[1][j] = cf_arr[1][j]+result_train[i][j]

    for i in range(2564,4233):
        for j in range(0,x):
            cf_arr[2][j] = cf_arr[2][j]+result_train[i][j]
            
    for i in range(4234,5328):
        for j in range(0,x):
            cf_arr[3][j] = cf_arr[3][j]+result_train[i][j]

    for i in range(5329,6395):
        for j in range(0,x):
            cf_arr[4][j] = cf_arr[4][j]+result_train[i][j]

    for i in range(6396,8352):
        for j in range(0,x):
            cf_arr[5][j] = cf_arr[5][j]+result_train[i][j]


    arr_sum = np.zeros(shape = (1, x))
    for i in range(0,x):
        for j in range(0,6):
            arr_sum[0][i] = arr_sum[0][i]+cf_arr[j][i]

    for i in range(1,x):
        for j in range(0,6):
            cf_arr[j][i] = cf_arr[j][i]/arr_sum[0][i]

    entropy_arr = np.zeros(shape = (1, x))

    for i in range(1,x):

        for j in range(0,6):
            if cf_arr[j][i]!=0:
                entropy_arr[0][i] = entropy_arr[0][i]+(cf_arr[j][i]*math.log(cf_arr[j][i]))
        if entropy_arr[0][i]!=0:
            entropy_arr[0][i] = (-1)*entropy_arr[0][i]

    
    token_arr = ["" for y in range(x)]
    counter = 0
    for key in token:
        token_arr[counter] = token_arr[counter] + key
        counter = counter+1

    entropy_arr = entropy_arr.transpose()

    entropy_arr = entropy_arr.tolist()

    elite_entropies = pd.DataFrame({'words':token_arr,'entropy':entropy_arr})

    elite_entropies.drop(0,axis=0,inplace=True)

    elite_entropies = elite_entropies.reset_index(drop=True)

    elite_entropies = elite_entropies.sort_values(by=['entropy'], ascending=True)
    elite_entropies = elite_entropies.reset_index(drop=True)

    elite_entropies.to_csv(f'{category}_entropy.csv')

    max_H = 0
    max_k = -1

    for i in range(2,len(elite_entropies['entropy'])-1):
        g_sum_1 = 0
        for j in range(0,i):
            g_sum_1 = g_sum_1 + elite_entropies.loc[j].at['entropy'][0]
        g_sum_2 = 0
        for j in range(i,len(elite_entropies['entropy'])):
            g_sum_2 = g_sum_2 + elite_entropies.loc[j].at['entropy'][0]

        g1_arr = np.zeros(i)
        g2_arr = np.zeros(len(elite_entropies['entropy'])-i)
        x = pow(10,-12)
        for j in range(0,i):
            g1_arr[j] = x
            if g_sum_1!=0:
                g1_arr[j] = x+(elite_entropies.loc[j].at['entropy'][0]/g_sum_1)

        count=0
        for j in range(i,len(elite_entropies['entropy'])):
            g2_arr[count] = x+(elite_entropies.loc[j].at['entropy'][0]/g_sum_2)
            count = count+1

        H1 = 0
        for j in range(0,i):
            H1 = H1+(g1_arr[j]*math.log(g1_arr[j]))

        H2 = 0
        for j in range(0,len(elite_entropies['entropy'])-i):
            H2 = H2+(g2_arr[j]*math.log(g2_arr[j]))

        H = -(H1+H2)

        if max_k==-1:
            max_H = H
            max_k = i

        else :
            max_H = max(max_H,H)
            if max_H==H:
                max_k = i


    print(category, max_k)
    return [category, max_k]

if __name__ == '__main__':
    train = pd.read_csv('train_resume_sorted.csv')
    train = train.dropna()
    train = train.drop(['ID'], axis=1)
    train = train.sort_values(by=['Category'],ascending=True)
    train = train.reset_index(drop=True)
    categories = train['Category'].unique()
    categories.sort()
    print(categories)
    train_list = [train for x in range(len(categories))]
    
    # csvs = [['Database_Administrator_6923.csv',6923 ], ['Network_Administrator_6145.csv', 6145], ['Project_manager_9213.csv', 9213], 
    #         ['Security_Analyst_5975.csv', 5975], ['Software_Developer_8475.csv', 8475], ['Systems_Administrator_8340.csv', 8340]]
    
    df_list = []
    num_elites  = []
    files = os.listdir('./')
    for category in categories:
        file_path = [x for x in files if category in x]
        df = pd.read_csv(file_path[0])
        df = df.iloc[:, 1:] if df.columns[0] == 'Unnamed: 0' else df 
        df_list.append(df)
        num_elite = file_path[0].split('_')
        num_elite = num_elite[-1]
        num_elite =  num_elite.replace('.csv', '')
        num_elites.append(int(num_elite))
        # print(category, num_elite)

    # acc,num_elite,category, train
    with multiprocessing.Pool() as pool:
        results = pool.starmap(unique_elite, zip(df_list,num_elites,categories, train_list))
    print(results)

    with open('unique_elite_key_word_extracted.txt', 'w') as file:
        for inner_list in results:
            file.write(' '.join(map(str, inner_list)) + '\n')
