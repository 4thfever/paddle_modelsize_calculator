# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Writer: Xinqiao Wang
Organization: Global Energy Interconnection Research Institute, SGCC
Date: 20210317
Objective: Import PaddlePaddle machine learning model (usually vision model)
            then give the evaluation of model back.
'''
from collections import OrderedDict
from prettytable import PrettyTable
from numpy import prod

def summary(main_prog, batch_size, bits_per_tensor):
    '''
    It can summary model's PARAMS, FLOPs until now.
    It support common operator like conv, fc, pool, relu, sigmoid, bn etc. 
    Args:
        main_prog: main program 
    Returns:
        print summary on terminal
    '''
    blocks = main_prog.blocks
    collected_ops_list = []
    unsupported = set()
    block_vars = {}
    for block in blocks:
        block_vars = {**block_vars, **block.vars}
    block_ops = [ele for block in blocks for ele in block.ops]
    # block_var: learnable variable，block_op:operator
    # 合并blocks（ops和vars并不严格对应,需要合并保证能搜索到所有）
    for one_op in block_ops:
        op_info = OrderedDict()
        spf_res = _summary_model(block_vars, one_op)
        if spf_res is None:
            continue
        if type(spf_res) == str:
            unsupported.add(one_op.type)
            continue
        # TODO: get the operator name
        op_info['type'] = one_op.type
        op_info['input_shape'] = spf_res[0][1:]
        op_info['out_shape'] = spf_res[1][1:]
        op_info['PARAMs'] = spf_res[2]
        op_info['FLOPs'] = spf_res[3]
        collected_ops_list.append(op_info)
    summary_table, total = _format_summary(collected_ops_list, batch_size, bits_per_tensor)
    _print_summary(summary_table, total, unsupported)


def _summary_model(block_vars, one_op):
    '''
    Compute operator's params and flops.
    Args:
        block_vars: all vars of one block
        one_op: one operator to count
    Returns:
        in_data_shape: one operator's input data shape
        out_data_shape: one operator's output data shape
        params: one operator's PARAMs 
        flops: : one operator's FLOPs
    '''
    if one_op.type in ['conv2d', 'depthwise_conv2d']:
        k_arg_shape = block_vars[one_op.input("Filter")[0]].shape
        in_data_shape = block_vars[one_op.input("Input")[0]].shape
        out_data_shape = block_vars[one_op.output("Output")[0]].shape
        c_out, c_in, k_h, k_w = k_arg_shape
        _, c_out_, h_out, w_out = out_data_shape
        assert c_out == c_out_, 'shape error!'
        k_groups = one_op.attr("groups")
        kernel_ops = k_h * k_w * (c_in / k_groups)
        bias_ops = 0 if one_op.input("Bias") == [] else 1
        params = c_out * (kernel_ops + bias_ops)
        flops = h_out * w_out * c_out * (kernel_ops + bias_ops)
        # base nvidia paper, include mul and add
        flops = 2 * flops

    elif one_op.type == 'pool2d':
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        _, c_out, h_out, w_out = out_data_shape
        k_size = one_op.attr("ksize")
        params = 0
        flops = h_out * w_out * c_out * (k_size[0] * k_size[1])

    elif one_op.type == 'mul':
        k_arg_shape = block_vars[one_op.input("Y")[0]].shape
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        # TODO: fc has mul ops
        # add attr to mul op, tell us whether it belongs to 'fc'
        # this's not the best way
        if 'fc' not in one_op.output("Out")[0]:
            return None
        k_in, k_out = k_arg_shape
        # bias in sum op
        params = k_in * k_out + 1
        flops = k_in * k_out

    elif one_op.type in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu']:
    # elif one_op.type in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu', 'elementwise_add', 'elementwise_mul', 'elementwise_div']:
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        params = 0
        if one_op.type == 'prelu':
            params = 1
        flops = 1
        for one_dim in in_data_shape:
            if one_dim != -1:
                # 如果不为-1
                flops *= one_dim

    elif one_op.type == 'batch_norm':
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Y")[0]].shape
        _, c_in, h_out, w_out = in_data_shape
        # gamma, beta
        params = c_in * 2
        # compute mean and std
        flops = h_out * w_out * c_in * 2

    else:
        # 有些没有被计算到的type，加入unsupported_set中
        # 某些操作，比如affine_channel，仅是仿射变换，不计入
        return one_op.type

    return in_data_shape, out_data_shape, params, flops


def _format_summary(collected_ops_list, batch_size, bits_per_tensor):
    '''
    Format summary report.
    Args:
        collected_ops_list: the collected operator with summary
    Returns:
        summary_table: summary report format
        total: sum param and flops
    '''

    summary_table = PrettyTable(
        ["No.", "TYPE", "INPUT", "OUTPUT", "PARAMs", "FLOPs"])
    summary_table.align = 'r'

    total = {}
    total_params = []
    total_flops = []
    total_outshape = []
    for i, one_op in enumerate(collected_ops_list):
        # notice the order
        table_row = [
            i,
            one_op['type'],
            one_op['input_shape'],
            one_op['out_shape'],
            int(one_op['PARAMs']),
            int(one_op['FLOPs']),
        ]
        if i == 0:
            input_shape = one_op['input_shape']
        summary_table.add_row(table_row)
        total_params.append(int(one_op['PARAMs']))
        total_flops.append(int(one_op['FLOPs']))
        total_outshape.append(one_op['out_shape'])

    total['params'] = total_params
    total['flops'] = total_flops
    total['out'] = total_outshape
    total['gpu'] = cal_gpu_memory(total, input_shape, batch_size, bits_per_tensor)
    return summary_table, total

def cal_gpu_memory(total, input_shape, batch_size, bits_per_tensor):
    gpu_input = prod(input_shape)
    gpu_param = total['params']
    gpu_backward_forward = sum([prod(ele) for ele in total['out']])
    gpu = (gpu_input + gpu_param + gpu_backward_forward)*(batch_size*bits_per_tensor/8) # bytes计数
    return gpu


def _print_summary(summary_table, total, unsupported):
    '''
    Print all the summary on terminal.
    Args:
        summary_table: summary report format
        total: sum param and flops
    '''
    parmas = total['params']
    flops = total['flops']
    gpu = total['gpu']
    print(
        "Notice: \n now supported ops include [Conv, DepthwiseConv, FC(mul), BatchNorm, Pool, Activation(sigmoid, tanh, relu, leaky_relu, prelu), Elementwise operations]"
    )
    print("Unsupported operator types:", unsupported)
    print(summary_table)
    print('Total PARAMs: {}({:.4f}M)'.format(
        sum(parmas), sum(parmas) / (10**6)))
    print('Total FLOPs: {}({:.2f}G)'.format(sum(flops), sum(flops) / 10**9))
    print('GPU Memory Usage: {}({:.2f}GB)'.format(sum(gpu), sum(gpu) / 10**9))
