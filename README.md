# Ŀ��/Objective

�ù��ߵ�Ŀ���Ƿ���ع���Paddle�����ѧϰģ�͵��Դ�����

This tool aims to provide a convenient way of evaluating the GPU comsumption of Paddle deep learning model

# ����/based on

�ù��߻���Paddle��model_stat�ļ���Torch��GPU�����ļ�

This tool is based on 
https://github.com/PaddlePaddle/Paddle/blob/8ff3550658e9fea3e652ebc2a34f62e54a59cd26/python/paddle/fluid/contrib/model_stat.py

https://github.com/jacobkimmel/pytorch_modelsize/blob/master/pytorch_modelsize.py

# ʹ��/Usage

```python
infer_prog = ...
summary(infer_prog, 16, 32)
+-----+---------+----------------+----------------+---------+------------+
| No. |    TYPE |          INPUT |         OUTPUT |  PARAMs |      FLOPs |
+-----+---------+----------------+----------------+---------+------------+
|   0 |  conv2d |  (3, 224, 224) | (64, 112, 112) |    9408 |  236027904 |
......
| 130 | sigmoid |   (81, 14, 14) |   (81, 14, 14) |       0 |      15876 |
+-----+---------+----------------+----------------+---------+------------+
Total PARAMs: 47931840(47.9318M)
Total FLOPs: 13681855940(13.68G)
GPU Memory Usage: 193644274688.0(193.64GB)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)# paddle_modelsize_calculator
# paddle_modelsize_calculator
