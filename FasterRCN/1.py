import numpy as np
# import tensorflow as tf

1.将类对应索引，做成一个字典
classes = ['car', 'cat', 'plante']
num_classes = 3
class_to_ind = dict(list(zip(classes, list(range(num_classes) ) ) ) )

2.对于标签数据可以使用pkl将数据保存，可提高读取效率
加载pkl
with open(pkl,'rb') as fid:
    roidb = pickle.load(fid)
    return roidb

写入pkl
with open(pkl,'wb') as fid:
    pickle.dump(gt_roidb,fid, pickle.HIGHEST_PROTOCOL)

3.读取pascol_voc标注信息（Annotation）
xml_filename = 'xxx.xml'图片#下面程序仅仅读取一张图片的标注信息
import xml.etree.ElementTree as ET
tree = ET.parse(xml_filename)
objs = tree.findall('object')
if not self.config['use_diff']:
    non_diff_objs = [
        obj for obj in objs if in(obj.find('difficult').text) == 0]
    #删选标签
    ]
    objs = non_diff_objs

num_objs = len(objes)
boxes = np.zeros((num_objs,4),dtype= np.uint16)
gc_classes = np.zeros((num_objs), dtype=np.int32)
overlaps = np.zeros( (num_objs, num_classes), dtype=np.float32 )
#overlaps 的每一行中对应的类别为1，其他为0
for ix, obj in enumerate(objs):
    bbox = obj.find('bndbox')
    x1 = float(bbox.find('xmin').text - 1)
    y1 = float(bbox.find('ymin').text - 1)
    x2 = float(bbox.find('xmax').text - 1)
    y2 = float(bbox.find('ymax').text - 1)
    cls = class_to_ind[obj.find('name').text.lower().strip()]
    boxes[ix,:] = [x1,y1,x2,y2]
    gt_classes[ix] = cls #类别对应的索引
    overlaps[ix,cls] = 1.0

    #overlaps,可以压缩
    #overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'boxes':boxes,
            'gt_classes':gt_classes,
            'gt_overlaps':overlaps,
            'flipped':False
            }

4.标注信息左右翻转
def append_flipped_images(self):
    for i in range(num_images):
        boxes = roidb[i]['boxes'].copy()
        oldx1 = boxes[:,0].copy()
        oldx2 = boxes[:,2].copy()
        #因为是左右翻转，因此y的值不会发生改变
        boxes[:,0] = widths[i] - oldx2 - 1
        boxes[:,2] = widths[i] - oldx1 - 1
        assert (boxes[:,2] >= boxes[:,0]).all()
        entry = {'boxes':boxes,
                 'gt_overlaps':roidb[i]['gt_overlaps'],
                 'gt_classes':roidb[i]['gt_classes'],
                 'flipped':True}
        roidb.append(entry)
        image_index = image_index * 2


