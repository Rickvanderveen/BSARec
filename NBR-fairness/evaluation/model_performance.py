from metrics import *
import pandas as pd
import json
import argparse
import os
from fair_metrics.Run_metrics_RecSys import metric_analysis as ma
from metric_utils.groupinfo import GroupInfo
import metric_utils.position as pos

def get_repeat_eval(pred_folder, dataset, size, file, number_list = None):
    history_file = f'../csvdata/{dataset}/{dataset}_train.csv'
    truth_file = f'../jsondata/{dataset}_future.json'
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    data_history = pd.read_csv(history_file)
    a_ndcg = []
    a_recall = []
    a_hit = []
    a_repeat_ratio = []
    a_explore_ratio = []
    a_recall_repeat = []
    a_recall_explore = []
    a_hit_repeat = []
    a_hit_explore = []

    # for ind in number_list:
    keyset_file = f'../keyset/{dataset}_keyset.json'
    # pred_file = f'{pred_folder}/{dataset}_pred{ind}.json'
    pred_file = f'{pred_folder}/{dataset}_pred.sjon'
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)
    
    ndcg = []
    recall = []
    hit = []
    repeat_ratio = []
    explore_ratio = []
    recall_repeat = []
    recall_explore = []
    hit_repeat = []
    hit_explore = []

    for user in keyset['test']: 
        pred = data_pred[user]
        truth = data_truth[user][1]
        # print(user)
        user_history = data_history[data_history['user_id'].isin([int(user)])]
        repeat_items = list(set(user_history['item_id']))
        truth_repeat = list(set(truth)&set(repeat_items)) # might be none
        truth_explore = list(set(truth)-set(truth_repeat)) # might be none

        u_ndcg = get_NDCG(truth, pred, size)
        ndcg.append(u_ndcg)
        u_recall = get_Recall(truth, pred, size)
        recall.append(u_recall)
        u_hit = get_HT(truth, pred, size)
        hit.append(u_hit)

        u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)# here repeat items
        repeat_ratio.append(u_repeat_ratio)
        explore_ratio.append(u_explore_ratio)

        if len(truth_repeat)>0:
            u_recall_repeat = get_Recall(truth_repeat, pred, size)# here repeat truth, since repeat items might not in the groundtruth
            recall_repeat.append(u_recall_repeat)
            u_hit_repeat = get_HT(truth_repeat, pred, size)
            hit_repeat.append(u_hit_repeat)

        if len(truth_explore)>0:
            u_recall_explore = get_Recall(truth_explore, pred, size)
            u_hit_explore = get_HT(truth_explore, pred, size)
            recall_explore.append(u_recall_explore)
            hit_explore.append(u_hit_explore)
        
    a_ndcg.append(np.mean(ndcg))
    a_recall.append(np.mean(recall))
    a_hit.append(np.mean(hit))
    a_repeat_ratio.append(np.mean(repeat_ratio))
    a_explore_ratio.append(np.mean(explore_ratio))
    a_recall_repeat.append(np.mean(recall_repeat))
    a_recall_explore.append(np.mean(recall_explore))
    a_hit_repeat.append(np.mean(hit_repeat))
    a_hit_explore.append(np.mean(hit_explore))
    #print(ind, np.mean(recall))
    #file.write(str(ind)+' '+str(np.mean(recall))+'\n')
   

    file.write('basket size: ' + str(size) + '\n')
    file.write('recall: '+ str([round(num, 4) for num in a_recall]) +' '+ str(round(np.mean(a_recall), 4)) +' '+ str(round(np.std(a_recall) / np.sqrt(len(a_recall)), 4)) +'\n')
    file.write('ndcg: '+ str([round(num, 4) for num in a_ndcg]) +' '+ str(round(np.mean(a_ndcg), 4)) +' '+ str(round(np.std(a_ndcg) / np.sqrt(len(a_ndcg)), 4)) +'\n')
    file.write('hit: '+ str([round(num, 4) for num in a_hit]) +' '+ str(round(np.mean(a_hit), 4)) +' '+ str(round(np.std(a_hit) / np.sqrt(len(a_hit)), 4)) +'\n')
    
    return np.mean(a_recall)


def get_eval_input(pred_folder, dataset, size, file, pweight, number_list = None): #add user_group is str 
    
    group_file = f'../methods/g-p-gp-topfreq/group_results/{dataset}_group_purchase.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    group_dict = dict()
    for name, item in group_item.items():
        group_dict[name] = len(item) #the number of each group
    group = GroupInfo(pd.Series(group_dict), 'unpop', 'pop', 'popularity') 

    IAA = []         
    EEL = []            
    EED = []             
    EER = []             
    DP = []           
    EUR = []          
    RUR = []       

    # for ind in number_list:
    keyset_file = f'../keyset/{dataset}_keyset.json'
    #keyset_file = f'../keyset/{dataset}_keyset_user.json'
    # pred_file = f'{pred_folder}/{dataset}_pred{ind}.json'
    # rel_file = f'{pred_folder}/{dataset}_rel{ind}.json'
    pred_file = f'{pred_folder}/{dataset}_pred.json'
    rel_file = f'{pred_folder}/{dataset}_rel.json'

    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)
    with open(rel_file, 'r') as f:
        data_rel = json.load(f)


    truth_file = f'../jsondata/{dataset}_future.json' # all users
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    rows = []
    for user_id, items in data_truth.items():
        #if user_id in keyset[user_group]: 
        if user_id in keyset['test']: #only evaluate test users
            for i, item_id in enumerate(items[1]):
                if str(item_id) in group_item['pop']:
                    rows.append((user_id, item_id, 1, 'pop', 1, 0))
                else:
                    rows.append((user_id, item_id, 1, 'unpop', 0, 1))
    test_rates = pd.DataFrame(rows, columns=['user', 'item', 'rating', 'popularity', 'pop', 'unpop']) 
    row = [] #relev 
    ros = [] #recs
    for user_id, items in data_pred.items():
        #if user_id in keyset[user_group]: 
        # user_id = user_id.split()[-1]
        if user_id in keyset['test']: #only evaluate test users
            for i, item_id in enumerate(items):
                if str(item_id) in group_item['pop']:
                    # print(f"data_rel[user_id] = {data_rel[user_id]} and item_id = {item_id}")
                    if item_id in data_rel[user_id]:
                        rel_value = 1
                    else:
                        rel_value = 0
                    # row.append((user_id, item_id, data_rel[user_id][item_id], 'pop', i+1))
                    row.append((user_id, item_id, rel_value, 'pop', i+1))
                    if item_id in data_truth[user_id][1]:
                        ros.append((user_id, item_id, i+1, 'pop', 1, 1, 0))
                    else:
                        ros.append((user_id, item_id, i+1, 'pop', 0, 1, 0))
                if str(item_id) in group_item['unpop']:
                    # print(f"data_rel[user_id] = {data_rel[user_id]} and item_id = {item_id}")
                    if item_id in data_rel[user_id]:
                        rel_value = 1
                    else:
                        rel_value = 0
                    # row.append((user_id, item_id, data_rel[user_id][item_id], 'unpop', i+1))
                    row.append((user_id, item_id, rel_value, 'unpop', i+1))
                    if item_id in data_truth[user_id][1]:
                        ros.append((user_id, item_id, i+1, 'unpop', 1, 0, 1))
                    else:
                        ros.append((user_id, item_id, i+1, 'unpop', 0, 0, 1))
    recs = pd.DataFrame(ros, columns=['user', 'item', 'rank', 'popularity', 'rating', 'pop', 'unpop']) 
    # print("Total 'pop' predictions:", recs[recs['popularity'] == 'pop'].shape[0])
    # print("Total 'unpop' predictions:", recs[recs['popularity'] == 'unpop'].shape[0])

    relev = pd.DataFrame(row, columns=['user', 'item', 'score', 'popularity', 'rank']) #in line with recs
    MA = ma(recs, test_rates, group, original_relev=relev)
    default_results = MA.run_default_setting(listsize=size, pweight=pweight)

    IAA.append(default_results['IAA'])       
    EEL.append(default_results['EEL'])     
    EED.append(default_results['EED'])       
    EER.append(default_results['EER'])       
    DP.append(default_results['logDP'])          
    EUR.append(default_results['logEUR'])          
    RUR.append(default_results['logRUR'])      

    #file.write('basket size: ' + str(size) + '\n')
    #file.write('weight:' + str(pweight) + '\n')

    #file.write(str(user_group) + '\n')
    file.write('IAA: ' + str([round(num, 4) for num in IAA]) +' '+ str(round(np.mean(IAA), 4)) +' '+ str(round(np.std(IAA) / np.sqrt(len(IAA)), 4)) +'\n')
    file.write('EEL: ' + str([round(num, 4) for num in EEL]) +' '+ str(round(np.mean(EEL), 4)) +' '+ str(round(np.std(EEL) / np.sqrt(len(EEL)), 4)) +'\n')
    file.write('EED: ' + str([round(num, 4) for num in EED]) +' '+ str(round(np.mean(EED), 4)) +' '+ str(round(np.std(EED) / np.sqrt(len(EED)), 4)) +'\n')
    file.write('EER: ' + str([round(num, 4) for num in EER]) +' '+ str(round(np.mean(EER), 4)) +' '+ str(round(np.std(EER) / np.sqrt(len(EER)), 4)) +'\n')
    file.write('DP: ' + str([round(num, 4) for num in DP]) +' '+ str(round(np.mean(DP), 4)) +' '+ str(round(np.std(DP) / np.sqrt(len(DP)), 4)) +'\n')
    file.write('EUR: ' + str([round(num, 4) for num in EUR]) +' '+ str(round(np.mean(EUR), 4)) +' '+ str(round(np.std(EUR) / np.sqrt(len(EUR)), 4)) +'\n')
    file.write('RUR: ' + str([round(num, 4) for num in RUR]) +' '+ str(round(np.mean(RUR), 4)) +' '+ str(round(np.std(RUR) / np.sqrt(len(RUR)), 4)) +'\n')
    
    
    return IAA
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, required=True, help='x')
    # parser.add_argument('--number_list', type=list, required=True, help='x')
    parser.add_argument('--method', type=str, required=True, help='x')
    parser.add_argument('--model', type=str, required=True, help='x')
    


    args = parser.parse_args()
    pred_folder = args.pred_folder
    method_name = args.method
    model_name = args.model
    if not os.path.exists('results/'):
        os.makedirs('results/')
    eval_file = f'results/eval_{method_name}_{model_name}.txt'
    f = open(eval_file, 'w')
    # for dataset in ['instacart', 'dunnhumby', 'tafeng']: 
    for dataset in ['LastFM']:
        f.write('############'+dataset+'########### \n')
        # get_repeat_eval(pred_folder, dataset, 10, number_list, f)
        # get_repeat_eval(pred_folder, dataset, 10, f)
        # get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default')
        get_eval_input(pred_folder, dataset, 10, f, pweight='default')

        '''
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default', user_group='test02')
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default', user_group='test24')
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default', user_group='test46')
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default', user_group='test68')
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight='default', user_group='test81')
        '''


        #for size in [10,20,30,40,50,60,70,80,90,100]:
            #get_eval_input(pred_folder, dataset, number_list, size, f, pweight='default')
        '''
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight=pos.geometric())
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight=pos.cascade())
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight=pos.logarithmic())
        get_eval_input(pred_folder, dataset, number_list, 10, f, pweight=pos.equality())
        '''
