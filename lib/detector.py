import onnxruntime
import cv2
import numpy as np

class Detector:
    def __init__(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider"])
        self.get_meta()
        self.K=100
        self.conf_thres = 0.4
        
    def get_meta(self):
        width = 1088
        height = 608
        inp_height = height
        inp_width = width
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        self.meta = {'c': c, 's': s,
                'out_height': inp_height // 4,
                'out_width': inp_width // 4}
        
    def post(self, X):        
        hm, wh, id_feature, reg = X
        id_feature /= np.linalg.norm(id_feature, ord=2, axis=1)
        # hm = sigmoid(hm)
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=self.K)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = np.squeeze(id_feature,axis=0)
        dets = post_process(dets, self.meta)
        dets = merge_outputs([dets], K=self.K)[1]
        remain_inds = dets[:, 4] > self.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        return dets, id_feature
    
    def forward(self, X):
        # x = np.random.randn(1,3,608,1088).astype(np.float32)
        output = self.session.run(None,{'0':X})
        # return output
        return self.post(output)

def sigmoid(x):
    np.warnings.filterwarnings('ignore', '(overflow|invalid)')
    return  1./(1.+ np.exp(-x))

def topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = max_pooling_forward(heat, (kernel, kernel), (1,1), (pad,pad))
    keep = (hmax == heat)
    return heat * keep

def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    # zero padding
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # out height and width
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z

def _topk(scores, K=40):
    batch, cat, height, width = scores.shape
    topk_scores, topk_inds = topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = np.true_divide(topk_inds, width).astype(np.int32).astype(np.float32)
    topk_xs   = (topk_inds % width).astype(np.int32).astype(np.float32)
      
    topk_score, topk_ind = topk(topk_scores.reshape(batch, -1), K)
    topk_clses = np.true_divide(topk_ind, K).astype(np.int32)
    topk_inds = _gather_feat(
        topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _tranpose_and_gather_feat(feat, ind):
    feat = np.transpose(feat, (0, 2, 3, 1)).copy()
    
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.shape[2]
    ind  = np.broadcast_to(np.expand_dims(ind, axis=2),(ind.shape[0], ind.shape[1], dim))
    feat = np.take_along_axis(feat, ind, axis=1)
    if mask is not None:
        mask = np.broadcast_to(np.expand_dims(mask, axis=2),feat.shape)
        feat = feat[mask]
        feat = feat.reshape(-1, dim)
    return feat

def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.shape

    # heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.reshape(batch, K, 4)
    else:
        wh = wh.reshape(batch, K, 2)
    clses = clses.reshape(batch, K, 1).astype(np.float32)
    scores = scores.reshape(batch, K, 1)
    if ltrb:
        bboxes = np.concatenate([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], axis=2)
    else:
        bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], axis=2)
    detections = np.concatenate([bboxes, scores, clses], axis=2)

    return detections, inds

def merge_outputs(detections, K=100):
    results = {}
    for j in range(1, 2):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, 2)])
    if len(scores) > K:
        kth = len(scores) - K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def post_process(dets, meta):
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 1)
    for j in range(1, 2):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]

def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
                dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
                dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]