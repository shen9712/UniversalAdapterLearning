# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import torch

# from . import BaseWrapperDataset, data_utils


# class AddTargetDataset(BaseWrapperDataset):
#     def __init__(
#         self,
#         dataset,
#         labels,  # dict, key=unit
#         pad,
#         eos,
#         batch_targets,
#         process_label=None,  # dict, key=unit
#         add_to_input=False,
#     ):
#         super().__init__(dataset)
#         self.labels = labels
#         self.batch_targets = batch_targets
#         self.pad = pad
#         self.eos = eos
#         self.process_label = process_label
#         self.add_to_input = add_to_input
        
#     def get_label(self, index):
#         result = {}
#         for key in self.labels.keys():
#             if self.process_label is None:
#                 result[key] = self.labels[key][index]
#             else:
#                 result[key] = self.process_label[key](self.labels[key][index])
#         return result  # dict
#         # return (
#         #     self.labels[index]
#         #     if self.process_label is None
#         #     else self.process_label(self.labels[index])
#         # )

#     def __getitem__(self, index):
#         item = self.dataset[index]
#         item["label"] = self.get_label(index)
#         return item  # 是一个dict

#     def size(self, index):
#         sz = self.dataset.size(index)
#         own_sz = len(self.get_label(index))
#         return (sz, own_sz)

#     def collater(self, samples):
#         collated = self.dataset.collater(samples)
#         if len(collated) == 0:
#             return collated
#         indices = set(collated["id"].tolist())
#         target = [s["label"] for s in samples if s["id"] in indices]  # 一个list，每个list中的元素是一个dict
#         target_tmp = {}
#         for key in target[0]:
#             target_tmp[key] = []
#         for ele in target:
#             for key, value in ele.items():
#                 target_tmp[key].append(value)  # 转换为一个dict，每一个dict对应的值是一个list

#         target = target_tmp
#         main_unit = list(target.keys())[0]
        
#         if self.batch_targets:
#             for key in self.labels.keys():  # szj
#                 collated["target_lengths" + key] = torch.LongTensor([len(t) for t in target[key]])
#                 # target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
#                 collated["target" + key] = data_utils.collate_tokens(target[key], pad_idx=self.pad, left_pad=False)
#                 collated["ntokens"] = collated["target_lengths" + main_unit].sum().item()
#         else:
#             collated["ntokens"] = sum([len(t) for t in target[main_unit]])


#         if self.add_to_input:
#             eos = target.new_full((target.size(0), 1), self.eos)
#             collated["target"] = torch.cat([target, eos], dim=-1).long()
#             collated["net_input"]["prev_output_tokens"] = torch.cat(
#                 [eos, target], dim=-1
#             ).long()
#             collated["ntokens"] += target.size(0)
#         return collated
