import pandas as pd
import pickle
import math
import numpy as np
import os
from heapq import nlargest
relevance_data =  pd.read_csv('../../data/mt_4272_2395_pctr_2024_TOIS.csv',header=None)
#Relevance score matrix, where each row represents a user's relevance score for an item
cr_mat = pd.read_csv('../../data/mt_j_sim_sym_01_TOIS_0.99.csv',header=None)
#A square matrix representing competitive relationships. cr_mat[i,j]=1 means that there is a competitive relationship between item i and item j.
cr_mat = cr_mat.to_numpy()
cr_mat =  (cr_mat+cr_mat.T)/2
item_id_list = list(range(relevance_data.shape[1]))
user_num,item_num = relevance_data.shape
top_k = 5
alpha = 0.025
exposure_total = 0#Exposure provided by a single ranked list
for _ in range(top_k):
    exposure_total += 1/math.log(_+2,2)
# start = time.time()
exposure_cum = np.squeeze(np.zeros((item_num,1)))#Record the cumulative exposure of each item
exposure_merit = np.zeros((item_num,1))#Exposure divided by merit
gradient_list = np.zeros((item_num,1))#Record the gradient at each iteration
nUtility = [] #nUtility at each time point
list_loss_cr = [] #loss at each time point
rec_list = dict() #Rank list after reordering
reg_coef = np.sum(cr_mat)
P_coef = alpha/reg_coef#Regularization coefficient of the Loss term
# time_start = time.time()  # 记录开始时间
for i_ in range(user_num):
    rel_temp = relevance_data.iloc[i_,:]#The relevance score of the current user
    rel_sort = nlargest(top_k,rel_temp)
    utility = 0
    for i in range(top_k):
        utility += rel_sort[i]/math.log(i+2,2)
    merit_temp = (relevance_data.iloc[0:i_+1,:].sum(axis=0)+np.ones(item_num))/(i_+1)#Prevent the risk of division by zero
    exposure_merit = exposure_cum/merit_temp

    #Computing Gradients
    for itemi in range(item_num):
        gradient_temp = (1-alpha)*rel_temp[itemi]
        cr_mat_temp = cr_mat[itemi]
        exposure_merit_sub = exposure_merit[itemi] - exposure_merit
        gradient_temp -= np.sum(cr_mat_temp*exposure_merit_sub)*2/merit_temp[itemi]*P_coef*2#修正乘以2
        gradient_list[itemi] = gradient_temp

    gradient_zip = zip(gradient_list,rel_temp,range(item_num))
    gradient_zip_sorted = nlargest(top_k,gradient_zip,key=lambda x:(x[0],x[1]))
    rec_list[i_] = []
    utility_re = 0
    for i in range(top_k):
        item_index = gradient_zip_sorted[i][2]
        rec_list[i_].append(item_index)
        utility_re += (rel_temp[item_index]/math.log(i+2,2))
        exposure_cum[item_index] += 1/math.log(i+2,2)
    nUtility.append(utility_re/utility)
    exposure_merit = exposure_cum/merit_temp
    exposure_merit_ = np.zeros((item_num,item_num))
    for itemi in range(item_num):
        exposure_merit_[itemi] = exposure_merit[itemi] - exposure_merit
    loss_cr = np.sum(exposure_merit_*exposure_merit_*cr_mat)
    list_loss_cr.append(loss_cr/reg_coef)
    print(f'第{i_}-th user，nUtility:{nUtility[-1]:.4f},cum_nUtility:{sum(nUtility)/len(nUtility):.4f},crloss:{list_loss_cr[-1]:.4f}')
os.makedirs(f"./GRCF_res/{alpha}",exist_ok=True)
with open(f"./GRCF_res/{alpha}/res_{alpha}_{top_k}.txt",'w') as f:
    f.write(f"耗时{time_sum:.2f}s,cum_nUtility:{avg_list(nUtility):.4f},L_d_2:{list_loss_cr[-1]:.4f}")
# with open(f"./GRCF_res/{alpha}/loss_div_1_list_{alpha}_{top_k}.list","wb") as f:
    # pickle.dump(Loss_div_1,f)
with open(f"./GRCF_res/{alpha}/loss_cr_list_{alpha}_{top_k}.list","wb") as f:
    pickle.dump(list_loss_cr,f)
with open(f"./GRCF_res/{alpha}/nUtility_list_{alpha}_{top_k}.list","wb") as f:
    pickle.dump(nUtility,f)
with open(f"./GRCF_res/{alpha}/exposure_merit_{alpha}_{top_k}.list","wb") as f:
    pickle.dump(exposure_merit,f)
with open(f"./GRCF_res/{alpha}/exposure_cum_{alpha}_{top_k}.list","wb") as f:
    pickle.dump(exposure_cum,f)
with open(f"./GRCF_res/{alpha}/rerank_rec_list_{alpha}_{top_k}.list","wb") as f:
    pickle.dump(rec_list,f)