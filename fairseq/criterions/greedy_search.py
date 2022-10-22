# # from math import log, exp
import torch


def batched_greedy_search(data, length):
    # INF = -1e10
    
    # data = data[0:length, :]  # b, t, d
    decoded = torch.argmax(data, dim=-1) # b, t
    
    result = []
    bsz = data.shape[0]
    for b in range(bsz):
        result_b = torch.unique_consecutive(decoded[b, 0:length[b]])
        result_b = result_b[torch.nonzero(result_b, as_tuple=True)]
        result.append(result_b)
    
    return result


if __name__ == '__main__':
######## 改进 用cuda加速
    import time
    
    data = [[0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.3, 0.2]]
    bsz = 1
    data = [data] * bsz
    length = [ 3 ] * bsz

    data = torch.log(torch.tensor(data))
    length = torch.tensor(length)
    # result = beam_search_decoder(data, length, k=10)
    print("****use greedy search decoder****")
    # for seq in result:
    #     print(seq)

    
    time_start = time.time() #开始计时
    
    result = batched_greedy_search(data, length, k=10)
    time_end = time.time()    #结束计时

    print('time cost', time_end - time_start, 's')
    
    
    for seq in result:
        print(seq)
        print('---')

# ([2], 0.241)
# ([1, 2], 0.229)
# ([1], 0.18)
# ([2, 1], 0.17499999999999996)
# ([1, 2, 1], 0.075)
# ([1, 1], 0.045000000000000005)
# ([2, 2], 0.023999999999999997)
# ([2, 1, 2], 0.015999999999999997)