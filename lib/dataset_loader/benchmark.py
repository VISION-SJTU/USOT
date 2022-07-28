from os.path import join, realpath, dirname, exists
import glob, json, os
import numpy as np

# ---------------------------------------
# Functions for testing on SOT benchmarks
# ---------------------------------------
def load_dataset(dataset):
    """
    Supported dataset: OTB2015, VOT2016, VOT2018, LaSOT, TrackingNet
    For testing VOT2020, you should use the vot-toolkit and the provided $USOT_PATH/scripts/test_vot2020.py
    """
    info = {}

    if 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        json_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            # path_name = info[v]['video_dir']
            info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = info[v]['video_dir']

    elif 'VOT' in dataset and (not 'VOT2019RGBT' in dataset) and (not 'VOT2020' in dataset):
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'color', 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'VOT2020' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = open(gt_path, 'r').readlines()
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'RGBT234' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        json_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['infrared_imgs'] = [join(base_path, path_name, 'infrared', im_f) for im_f in
                                        info[v]['infrared_imgs']]
            info[v]['visiable_imgs'] = [join(base_path, path_name, 'visible', im_f) for im_f in
                                        info[v]['visiable_imgs']]
            info[v]['infrared_gt'] = np.array(info[v]['infrared_gt'])  # 0-index
            info[v]['visiable_gt'] = np.array(info[v]['visiable_gt'])  # 0-index
            info[v]['name'] = v

    elif 'VOT2019RGBT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            in_image_path = join(video_path, 'ir', '*.jpg')
            rgb_image_path = join(video_path, 'color', '*.jpg')
            in_image_files = sorted(glob.glob(in_image_path))
            rgb_image_files = sorted(glob.glob(rgb_image_path))

            assert len(in_image_files) > 0, 'please check RGBT-VOT dataloader'
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'infrared_imgs': in_image_files, 'visiable_imgs': rgb_image_files, 'gt': gt, 'name': video}

    elif 'VISDRONEVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'VISDRONETEST' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'initialization')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10KVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10K' in dataset:  # GOT10K TEST
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            if 'json' in video:
                continue
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}

    elif 'TRACKINGNET' in dataset:  # TRACKINGNET TEST
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        seq_path = join(base_path, "frames")

        videos = sorted(os.listdir(seq_path))
        videos = [v_name for v_name in videos if not v_name.endswith(".json")]
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            image_files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
            gt_path = join(seq_path, "..", "anno", '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}

    elif 'LASOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset)
        json_path = join(realpath(dirname(__file__)), '../../datasets_test', dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        testingvideos = list(jsons.keys())
        father_videos = sorted(os.listdir(base_path))
        for f_video in father_videos:
            if f_video not in testingvideos:
                continue
            f_video_path = join(base_path, f_video)
            # ground truth
            gt_path = join(f_video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            gt = gt - [1, 1, 0, 0]
            # get img file
            img_path = join(f_video_path, 'img', '*jpg')
            image_files = sorted(glob.glob(img_path))

            info[f_video] = {'image_files': image_files, 'gt': gt, 'name': f_video}

    elif 'DAVIS' in dataset and 'TEST' not in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), '../../datasets_test', 'DAVIS', 'ImageSets', dataset[-4:],
                         'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video

    elif 'YTBVOS' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../datasets_test', 'YTBVOS', 'valid')
        json_path = join(realpath(dirname(__file__)), '../../datasets_test', 'YTBVOS', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f + '.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)

    else:
        raise ValueError("Dataset not support now, edit for other dataset youself...")

    return info
