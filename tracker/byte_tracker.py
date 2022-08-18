import numpy as np
from collections import deque
import os
import os.path as osp
import copy

# import scipy.spatial.distance
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from .kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, clss, depth):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.depth = depth
        self.clss_pool = np.full(15, 35)
        self.n = -1
        self.clss = int(clss)
        self.live_frame = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
            #     print(f'stracks[{i}].mean: {stracks[i].mean}')
            #     print(f'stracks[{i}].covariance: {stracks[i].covariance}')
            # print(f'stracks:{stracks}')

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.depth = new_track.depth
        self.n = (self.n + 1) % 15
        self.clss_pool[self.n] = new_track.clss
        self.clss = np.bincount(self.clss_pool).argmax()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.depth = new_track.depth
        self.n = (self.n + 1) % 15
        self.clss_pool[self.n] = new_track.clss
        self.clss = np.bincount(self.clss_pool).argmax()

        # if self.clss != 35:
        #     self.track_id = self.next_id()

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.stable_tracked_strackes = []  # type: # list[STrack]

        # self.stable_id = 0

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        clsses = None
        depths = None

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5] * (1 - output_results[:, 6] / 255)
            # scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            clsses = output_results[:, 5]
            # scores = 1 - output_results[:, 6]/255
            depths = output_results[:, 6]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        depths_second = depths[inds_second]
        scores_second = scores[inds_second]
        dets = bboxes[remain_inds]
        depths_keep = depths[remain_inds]
        scores_keep = scores[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, d) for
                          (tlbr, s, c, d) in zip(dets, scores_keep, clsses, depths_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # print(f'dists: {dists}')
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # print(f'u_track:{u_track}')
        # print(f'u_detection:{u_detection}')
        # print(f'matches:{matches}')
        # print('-----------------------------------------------')
        u_track = list(u_track)
        u_detection = list(u_detection)
        class_filtered_matches = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # print(f'track.clss:{track.clss}')
            # print(f'det.clss:{det.clss}')
            if track.clss != 35:
                if track.clss == det.clss:
                    class_filtered_matches.append([itracked, idet])
                else:
                    u_track.append(itracked)
                    u_detection.append(idet)
            else:
                class_filtered_matches.append([itracked, idet])

        matches = class_filtered_matches
        # print(f'u_track:{u_track}')
        # print(f'u_detection:{u_detection}')
        # print(f'matches:{matches}')
        # print('------------------------------------------------')

        depth_filtered_matches = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # print(f'track.depth:{track.depth}')
            # print(f'det.depth:{det.depth}')
            if abs(track.depth - det.depth) < 20:
                depth_filtered_matches.append([itracked, idet])
            else:
                u_track.append(itracked)
                u_detection.append(idet)
        matches = depth_filtered_matches
        u_track = np.array(u_track)
        u_detection = np.array(u_detection)

        # print(f'u_track:{u_track}')
        # print(f'u_detection:{u_detection}')
        # print(f'matches:{matches}')
        # print('=======================================================')

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, d) for
                                 (tlbr, s, c, d) in zip(dets_second, scores_second, clsses, depths_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # self.stable_tracked_strackes = [track for track in self.tracked_stracks if track.clss != 35]

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


class StableTrack:
    def __init__(self, tlbr, score, clss, depth, id):
        self.id = id
        self.count = 0
        self.tlbr = tlbr
        self.score = score
        self.clss = clss
        self.depth = depth
        self.disappear = 0
        self.appeared = 0
        self.is_alive = True
        self.img_save = False

    def update(self, new_track):
        self.tlbr = new_track.tlbr
        self.score = new_track.score
        self.depth = new_track.depth
        self.appeared += 1
        self.disappear = 0

    def __repr__(self):
        return 'OT_{:4s} ({:4d})'.format(str(self.id), self.appeared)


class StableTracker:
    def __init__(self):
        self.stable_stracks = []  # type: list[StableTrack]
        self.stable_id = 0
        self.stable_cnt = 0

    def update(self, online_targets):
        if self.stable_cnt:
            # for target in online_targets:
            live_tracks = []
            clsses = set(t.clss for t in online_targets)
            for track in self.stable_stracks:
                if track.is_alive:
                    live_tracks.append(track)
                    clsses.add(track.clss)
            # live_tracks = [t for t in self.stable_stracks if t.is_alive]
            print(clsses)
            # candidate_each_class = []
            for c in clsses:
                print(f'current class : {c}')
                lt = [t for t in live_tracks if t.clss == c]
                ot = [t for t in online_targets if t.clss == c]
                ltsd = [[t.score, t.depth, (t.tlbr[0] + t.tlbr[2]) // 2, (t.tlbr[1] + t.tlbr[3]) // 2] for t in lt]
                otsd = [[t.score, t.depth, (t.tlbr[0] + t.tlbr[2]) // 2, (t.tlbr[1] + t.tlbr[3]) // 2] for t in ot]

                print(f'lt: {lt}')
                print(f'ot: {ot}')

                if not len(lt):
                    print('No lt')
                    for target in ot:
                        self.stable_cnt += 1
                        self.stable_id += 1
                        self.stable_stracks.append(
                            StableTrack(target.tlwh, target.score, target.clss, target.depth, self.stable_id))

                if not len(ot):
                    print('No ot')
                    for track in lt:
                        track.disappear += 1
                        if track.disappear > 30:  # disappear limit
                            track.is_alive = False
                            self.stable_cnt -= 1

                if not len(lt) or not len(ot):
                    continue

                D = cdist(ltsd, otsd)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                print(f'D.min() : {D.min()}')
                print(f'D.max() : {D.max()}')

                usedRows = set()
                usedCols = set()

                for row, col in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue

                    # print(f'D : {D}')

                    print(f'D[row]:{D[row]}')
                    print(f'D[row][col] : {D[row][col]}')
                    # if D[row][col] > 600:
                    #     continue
                    # print(f'D[col]:{D[col]}')

                    lt[row].update(ot[col])

                    usedRows.add(row)
                    usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        print(f'{lt[row]} is disappeared {lt[row].disappear}')
                        lt[row].disappear += 1
                        if lt[row].disappear > 30:  # disappear limit
                            lt[row].is_alive = False
                            self.stable_cnt -= 1
                else:
                    for col in unusedCols:
                        target = ot[col]
                        self.stable_cnt += 1
                        self.stable_id += 1
                        self.stable_stracks.append(
                            StableTrack(target.tlwh, target.score, target.clss, target.depth, self.stable_id))
                        # self.register(b_data[col], s_data[col], cls_data[col], conf_data[col])

            ######################## Matching...



            # dists = matching.iou_distance(live_tracks, online_targets)
            # dists = matching.fuse_score(dists, online_targets)
            # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.1)
            # print(f'matches: {matches}')
            # print(f'u_track: {u_track}')
            # print(f'u_detection: {u_detection}')

            # for itracked, idet in matches:
            #     track = self.stable_stracks[itracked]
            #     det = online_targets[idet]
            #     track.update(det)
            #
            # for itrack in u_track:
            #     track = self.stable_stracks[itrack]
            #     track.disappear += 1
            #     if track.disappear > 30:
            #         track.is_alive = False
            #         self.stable_cnt -= 1
            #
            # for idet in u_detection:
            #     target = online_targets[idet]
            #     self.stable_cnt += 1
            #     self.stable_id += 1
            #     self.stable_stracks.append(
            #         StableTrack(target.tlwh, target.score, target.clss, target.depth, self.stable_id))

        else:
            for target in online_targets:
                self.stable_cnt += 1
                self.stable_id += 1
                self.stable_stracks.append(
                    StableTrack(target.tlwh, target.score, target.clss, target.depth, self.stable_id))

        return [i for i in self.stable_stracks if i.is_alive and i.disappear == 0]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
