# from math import log, exp
import numpy as np


def beam_search_decoder(lis):
    data, length, k = lis
    sequences = [[list(), 0.0]]  ## log(1) = 0
    # print(length)
    # print(data.shape)
    
    for row_i in range(length):
        row = data[row_i]
        
        row_index = zip(row, range(len(row)))  # 一列tuple，(score，label)
        sorted_row = sorted(row_index, key=lambda x: x[0], reverse=True)  # 将该帧的每个节点的概率从高到低排序
        k_2 = k*k if k*k < len(sorted_row) else len(sorted_row)
        sorted_row = sorted_row[: k_2]  # 选择前k2个概率最高的节点
        
    # for row in data: # 遍历每一帧
        all_candidates = list()

        for i in range(len(sequences)):  # 扩展每一个beam
            seq, score = sequences[i]
            # k_2 = k*k if k*k < len(row) else len(row)
            # for j in range(k_2):  # 只考虑top_k2个标签
            for row, index in sorted_row:
                # candidate = [seq + [j], score * -log(row[j])]
                # candidate = [seq + [j], score + row[j]]  # 取对数后直接相加
                candidate = [seq + [index] , score + row]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)  # 按score从大到小排序
        k_2 = k*k if k*k < len(ordered) else len(ordered)
        sequences = ordered[:k_2]  # 选择前k2个最好的
    
    # for seq in sequences:
    #     print(seq)

    # 进行B映射
    merged_seq = list()
    merged_score = list()
    
    for candidate in sequences:
        # print(candidate)
        seq = B_map(candidate[0])
        # print(seq)
        score = np.exp(candidate[1])  # 恢复为概率
        if seq != []:
            if seq not in merged_seq:
                merged_seq.append(seq)
                merged_score.append(score)
            else:
                idx = merged_seq.index(seq)
                merged_score[idx] = merged_score[idx] + score
            
    # 排序
    all_candidates = zip(merged_seq, merged_score)
    ordered_score = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)  # 按score从大到小排序

    sequences = ordered_score[:k]  # 选择前k个最好的
    return sequences


# def greedy_decoder(data):

#     return [np.argmax(s) for s in data]

def B_map(seq, blank = 0):
    result = list()
    for i in range(len(seq)-1):
        if seq[i] != blank and seq[i] != seq[i+1]:
            result.append(seq[i])
    if seq[-1] != blank:
        result.append(seq[-1])
    return result

def batched_beam_search_para(data, length, k):
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(16)
    
    # print(data.shape)
    results = []
    bsz = len(data)
    # for b in range(bsz):
    #     lis = [data[b, ...], length[b], k]
    #     results.append(beam_search_decoder(lis))
    
    lis = []
    for b in range(bsz):
        lis.append([data[b, ...], length[b], k])

    time_start = time.time() #开始计时
    results = pool.map(beam_search_decoder, zip(data, length, [k]*bsz))
    time_end = time.time()    #结束计时
    print('time cost para', time_end - time_start, 's')
    pool.close()
    pool.join()
    
    
    return results

def batched_beam_search(data, length, k):
    
    # print(data.shape)
    results = []
    bsz = len(data)
    time_start = time.time() #开始计时
    for b in range(bsz):
        lis = [data[b, ...], length[b], k]
        results.append(beam_search_decoder(lis))
    time_end = time.time()    #结束计时
    print('time cost seri', time_end - time_start, 's')
    
    
    return results


if __name__ == '__main__':
######## 改进 用cuda加速
    import time
    
    data = [[0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.3, 0.2]]
    data = np.random.randint(1,100,size=(3000,100))  # t, d
    data = (data.T/(data.T.sum(axis=0))).T
    # exit(2)
    data = np.log(np.array(data))
    
    bsz = 30
    length = [ 3 ] * bsz
    length = np.array(length)
    # result = beam_search_decoder(data, length, k=10)
    print("****use beam search decoder****")
    # for seq in result:
    #     print(seq)

    data = [data] * bsz
    data = np.array(data)
    
    
    
    result = batched_beam_search_para(data, length, k=10)
    result = batched_beam_search(data, length, k=10)
    


    # for b in range(len(data)):
    #     for seq in result[b]:
    #         print(seq)
    #     print('---')
        
    

# ([2], 0.241)
# ([1, 2], 0.229)
# ([1], 0.18)
# ([2, 1], 0.17499999999999996)
# ([1, 2, 1], 0.075)
# ([1, 1], 0.045000000000000005)
# ([2, 2], 0.023999999999999997)
# ([2, 1, 2], 0.015999999999999997)