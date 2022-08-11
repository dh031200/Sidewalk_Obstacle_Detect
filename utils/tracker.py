from collections import OrderedDict
from scipy.spatial import distance as dist
from scipy.stats import mode
import numpy as np
import sys
import cv2

class Surface_tracker:
    def __init__(self, width, height, disappear_limit=0):
        self.width = width
        self.height = height
        self.segs = OrderedDict()
        self.centroid = OrderedDict()
        self.clsses = OrderedDict()
        self.confs = OrderedDict()
        self.canvas = self.reset_canvas()
        self.disappear = np.full((self.height, self.width), 0, dtype=np.uint64)
        # self.disappear = np.full((self.height, self.width,5), 0, dtype=np.uint64)
        self.disappear_limit = disappear_limit
        self.n = 0
        
    
    # def update(self, result):
    #     # self.canvas = self.reset_canvas()
    #     segs = result[1]
    #     self.indices = separate(all_bboxes=result[0], all_segs=segs, conf=0.1, surface=True)
    #     self.disappear += 1
    #     canvas = self.reset_canvas()
    #     for i in self.indices:
    #         canv = np.full((self.height, self.width), False)
    #         for j in self.indices[i]:
    #             canv = np.bitwise_or(canv, segs[i][j])
    #         # nz = list(canv.nonzero())
    #         # self.history[tuple(nz + [np.array([self.n % self.disappear_limit]*len(nz[0]))])]
    #         self.history[canv.nonzero(),self.n % self.disappear_limit] = i
    #         # canvas[canv.nonzero()] = i
    #         self.disappear[canv.nonzero()] = 0
    #     self.canvas = np.array([[mode(j)[0][0] for j in i] for i in self.history])
    #     # self.canvas = mode(self.history,axis=2)[0].reshape(-1,self.height,self.width)
    #     self.n += 1
    #     # self.canvas[np.where(self.disappear >= self.disappear_limit)] = 255
        
    def update(self, result):
        # self.canvas = self.reset_canvas()
        segs = result[1]
        # self.indices = separate(all_bboxes=result[0], all_segs=segs, conf=0.1, surface=True)
        # self.disappear += 1
        # for i in self.indices:
        #     # self.disappear[:,:,self.n] =
        #     canv = np.full((self.height, self.width), False)
        #     for j in self.indices[i]:
        #         canv = np.bitwise_or(canv, segs[i][j])
        #     self.canvas[canv.nonzero()] = i
        #     self.disappear[canv.nonzero()] = 0
        # self.canvas[np.where(self.disappear >= self.disappear_limit)] = 255

        self.indices = separate(all_bboxes=result[0], all_segs=segs, conf=0.1, surface=True)
        self.disappear += 1
        for i in self.indices:
            canv = np.full((self.height, self.width), False)
            for j in self.indices[i]:
                canv = np.bitwise_or(canv, segs[i][j])
            self.canvas[canv.nonzero()] = i
            self.disappear[canv.nonzero()] = 0
        self.canvas[np.where(self.disappear >= self.disappear_limit)] = 255
        
            
    def reset_canvas(self):
        return np.full((self.height, self.width), 255, dtype=np.uint8)
    
        

class Tracker:
    def __init__(self, disappear_limit=0):
        self.obj_ID = 0
        self.cnt = 0
        self.disappear_limit = disappear_limit
        self.bboxes = OrderedDict()
        self.segs = OrderedDict()
        self.centroid = OrderedDict()
        self.clsses = OrderedDict()
        self.confs = OrderedDict()
        self.disappear = OrderedDict()
        self.frame = None
        self.history = []
        self.history_img = []


    def process(self, b_data, s_data, cls_data, conf_data):
        if self.cnt == 0:
            for b, s, cl, cf in zip(b_data, s_data, cls_data, conf_data):
                self.register(b, s, cl, cf)

        else:
            new_centroids = np.zeros((len(b_data), 2), dtype="int")
            for i in range(len(b_data)):
                new_centroids[i] = self.centroider(b_data[i])

            IDs = list(self.centroid.keys())
            Centroids = list(self.centroid.values())

            D = dist.cdist(np.array(Centroids), new_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                ID = IDs[row]

                if self.clsses[ID] != cls_data[col]:
                    ID = None
                    continue

                self.bboxes[ID] = b_data[col]
                self.segs[ID] = s_data[col]
                self.centroid[ID] = new_centroids[col]
                self.confs[ID] = conf_data[col]
                self.disappear[ID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    ID = IDs[row]
                    self.disappear[ID] += 1
                    if self.disappear[ID] > self.disappear_limit:
                        self.deregister(ID)
            else:
                for col in unusedCols:
                    self.register(b_data[col], s_data[col], cls_data[col], conf_data[col])
                    

    def update(self, result, img):
        self.frame = img
        # OD Data processing
        b_data, s_data, cls_data, conf_data = separate(all_bboxes=result[0], all_segs=result[1],
                                                                   conf=0.4)
        # Update tracker
        if self.cnt > 0 and len(b_data) == 0:
            for i in self.disappear:
                self.disappear[i] += 1
        else:
            self.process(b_data, s_data, cls_data, conf_data)


    def centroider(self, bbox):
        return [int((bbox[0] + bbox[2]) / 2.0), int((bbox[1] + bbox[3]) / 2.0)]

    def register(self, b, s, cl, cf):
        obj_ID = self.obj_ID
        self.clsses[obj_ID] = cl
        self.bboxes[obj_ID] = b
        self.segs[obj_ID] = s
        self.centroid[obj_ID] = self.centroider(b)
        self.confs[obj_ID] = cf
        self.disappear[obj_ID] = 0
        self.obj_ID += 1
        self.cnt += 1

    def deregister(self, idx):
        # self.history.append(self.clsses[idx])
        # xmin, ymin, xmax, ymax = self.bboxes[idx]
        # cv2.imwrite(f'result/imgs/test_{idx}.png', self.frame[ymin:ymax,xmin:xmax])
        # self.history_img.append(self.frame[ymin:ymax,xmin:xmax])

        # cv2.imwrite(f'result/imgs/test_{idx}.png', self.history_img[idx])
        # print(self.bboxes[idx])
        del self.bboxes[idx]
        del self.segs[idx]
        del self.clsses[idx]
        del self.centroid[idx]
        del self.confs[idx]
        del self.disappear[idx]
        self.cnt -= 1


def separate(all_bboxes, all_segs, conf, surface=False):
    filtered_indice = OrderedDict()
    for clss in range(len(all_bboxes)):
        bboxes = all_bboxes[clss]
        conf_filtered_idx = []
        for j in range(len(bboxes)):
            if bboxes[j][4] > conf:  # and (((bboxes[j][2]-bboxes[j][0])*(bboxes[j][3]-bboxes[j][1])) > 5000):
                conf_filtered_idx.append(j)

        # conf_filtered_idx = [j for j in range(len(bboxes)) if bboxes[j][4] > conf]
        if len(conf_filtered_idx) > 0:
            filtered_indice[clss] = conf_filtered_idx
    if surface:
        return filtered_indice        

    b_data = []
    s_data = []
    cls_data = []
    conf_data = []
    for clss in filtered_indice:
        bboxes = all_bboxes[clss]
        segs = all_segs[clss]
        for idx in filtered_indice[clss]:
            b_data.append(list(map(int, bboxes[idx][:4])))
            s_data.append(segs[idx])
            cls_data.append(clss)
            conf_data.append(bboxes[idx][4])

    return b_data, s_data, cls_data, conf_data

#     def process(self, b_data, s_data, cls_data, conf_data):
#         update_list = [-1]*len(s_data)
#         if self.cnt != 0:
#             update_list = self.comparison(update_list, b_data, cls_data)

#         for u, b, s, cl, cf in zip(update_list, b_data, s_data, cls_data, conf_data):
#             self.register(u, b, s, cl, cf)

#         disappeared_list = []
#         for i in self.bboxes:
#             self.disappear[i] += 1
#             if self.disappear[i] > self.disappear_limit:
#                 disappeared_list.append(i)
#         for i in disappeared_list:
#             self.deregister(i)

#     def comparison(self, update_list, bboxes, clsses):
#         new_centroids = np.zeros((len(bboxes), 2), dtype="int")
#         for (i, (min_x, min_y, max_x, max_y)) in enumerate(bboxes):
#             cX = int((min_x + max_x) / 2.0)
#             xY = int((min_y + max_y) / 2.0)
#             new_centroids[i] = (cX,cY)

#         IDs = list(self.centroid.keys())
#         Centroids = list(self.centroid.values())
#         print(new_centroids)


#         D = dist.cdist(np.array(Centroids), new_centroids)

#         rows = D.min(axis=1).argsort()
#         cols = D.argmin(axis=1)[rows]

#         usedRows = set()
#         usedCols = set()

#         for row, col in zip(rows, cols):
#             if row in usedRows or col in usedCols:
#                 continue

#             ID = IDs[row]


# def register(self, u, b, s, cl, cf):
#     if u == -1:
#         obj_ID = self.obj_ID
#         self.obj_ID += 1
#         self.clsses[obj_ID] = cl
#         self.cnt += 1
#     else:
#         obj_ID = u
#         if self.confs[obj_ID] < cf:
#             self.clsses[obj_ID] = cl
#     self.bboxes[obj_ID] = b
#     self.segs[obj_ID] = s
#     self.centroid[obj_ID] = centroider(b)
#     self.confs[obj_ID] = cf
#     self.disappear[obj_ID] = 0


''' # process for Option 1, 2
    def process(self, b_data, s_data, cls_data, conf_data):
        update_list = [-1]*len(s_data)
        if self.cnt != 0:
            update_list = self.comparison(update_list, b_data, s_data, cls_data)
            
        for u, b, s, cl, cf in zip(update_list, b_data, s_data, cls_data, conf_data):
            self.register(u, b, s, cl, cf)
            
        disappeared_list = []
        for i in self.bboxes:
            self.disappear[i] += 1
            if self.disappear[i] > self.disappear_limit:
                disappeared_list.append(i)
        for i in disappeared_list:
            self.deregister(i)
'''

'''        
    def comparison(self, update_list, bboxes, segs, clsses): # Option 1 : using Segmentation and Bbox [not finished]
        similarity_list = [0]*len(segs)
        for new in range(len(segs)):
            n_x_min, n_y_min, n_x_max, n_y_max = bboxes[new]
            new_seg = segs[new][n_y_min:n_y_max,n_x_min:n_x_max]
            for idx in self.segs:
                if self.clsses[idx] != clsses[new]:
                    continue
                p_x_min, p_y_min, p_x_max, p_y_max = self.bboxes[idx]
                previous_seg = self.segs[idx][p_y_min:p_y_max,p_x_min:p_x_max]
                
                w_diff = max(new_seg.shape[1], previous_seg.shape[1]) - min(new_seg.shape[1], previous_seg.shape[1])
                if new_seg.shape[1] > previous_seg.shape[1]:
                    previous_seg = np.pad(previous_seg, ((0,0),(0,w_diff)), 'constant', constant_values=False)
                elif new_seg.shape[1] < previous_seg.shape[1]:
                    new_seg = np.pad(new_seg, ((0,0),(0,w_diff)), 'constant', constant_values=False)

                h_diff = max(new_seg.shape[0], previous_seg.shape[0]) - min(new_seg.shape[0], previous_seg.shape[0])
                if new_seg.shape[0] > previous_seg.shape[0]:
                    previous_seg = np.pad(previous_seg, ((0,h_diff),(0,0)), 'constant', constant_values=False)
                elif new_seg.shape[0] < previous_seg.shape[0]:
                    new_seg = np.pad(new_seg, ((0,h_diff),(0,0)), 'constant', constant_values=False)
                
                # previous_seg = self.segs[idx]
                similarity = (previous_seg & new_seg).sum()
                if int(previous_seg.sum()*0.7) < similarity < int(previous_seg.sum()*1.3):
                    if abs(previous_seg.sum() - similarity_list[new]) > abs(previous_seg.sum() - similarity):
                        update_list[new] = idx
                        similarity_list[new] = similarity
                    break
        return update_list
'''

'''
    def comparison(self, update_list, bboxes, segs):  # Option 2 : using only Segmentation [not finished]
    similarity_list = [0]*len(segs)
    for new in range(len(segs)):
        n_x_min, n_y_min, n_x_max, n_y_max = bboxes[new]
        new_seg
        for idx in self.segs:
            p_x_min, p_y_min, p_x_max, p_y_max = self.bboxes[idx]
            previous_seg = self.segs[idx][p_y_min:p_y_max,p_x_min:p_x_max]
            # previous_seg = self.segs[idx]
            similarity = (previous & segs[new]).sum()
            if int(previous.sum()*0.3) < similarity < int(previous.sum()*1.7):
                if abs(previous.sum() - similarity_list[new]) > abs(previous.sum() - similarity):
                    update_list[new] = idx
                    similarity_list[new] = similarity
                break
    return update_list
'''
