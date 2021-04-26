
import paddle
from paddle import fluid
from model_stat import summary

if __name__ == '__main__':
    import sys
    sys.path.append(r"C:\\Users\\wang\\PaddleDetection")
    from ppdet.modeling import (MaskRCNN, ResNet, ResNetC5, RPNHead, RoIAlign,
                            BBoxHead, MaskHead, BBoxAssigner, MaskAssigner)
    roi_size = 14

    model = MaskRCNN(
    ResNet(feature_maps=4),
    RPNHead(),
    BBoxHead(ResNetC5()),
    BBoxAssigner(),
    RoIAlign(resolution=roi_size),
    MaskAssigner(),
    MaskHead())

    h = 224
    w = 224
    feed_var_def = [
        {'name': 'image',  'shape': (16, 3, h, w)},
        {'name': 'im_info',  'shape': [16, 3]},
        {'name': 'im_shape', 'shape': [16, 3]},
    ]

    paddle.enable_static()
    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            feed_vars = {
                var['name']: fluid.data(
                    name=var['name'],
                    shape=var['shape'],
                    dtype='float32',
                    lod_level=0) for var in feed_var_def
            }
            test_fetches = model.test(feed_vars)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    summary(infer_prog, batch_size=16, bits_per_tensor=32)
