import sys,os
import json
import glob
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import ipdb
import numpy as np
import random



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# Get crop position of hand and keep the area is square.
def getRecFromKeypoints(hand_pts, factor):
    xmin,xmax,ymin,ymax = hand_pts[0][0], hand_pts[0][0], hand_pts[0][1], hand_pts[0][1] 
    for point in hand_pts:
        x,y = point[:2]
        xmin = x if x < xmin else xmin
        ymin = y if y < ymin else ymin
        xmax = x if x > xmax else xmax
        ymax = y if y > ymax else ymax
    
    
    cx, cy = (xmax+xmin)//2, (ymax+ymin)//2
    width, height = (xmax-xmin), (ymax-ymin)

    # scale.
    scale = 1.5
    width, height = int(scale * max(width, height)), int(scale * max(width, height))

    # crop. 
    factor_x, factor_y = factor
    offset_x = width * (0.00 + random.random()*0.1) # 0~0.25
    offset_y = height * (0.00 + random.random()*0.1)

    cx = cx + offset_x * factor_x
    cy = cy + offset_y * factor_y
        
    return (cx,cy,width,height)


phase = 'val'

savedir = './data-crop/imgs-{}/'.format(phase)
savedir_labeled = './data-crop/imgs-{}-labeled/'.format(phase)

if not os.path.exists(savedir):
    os.makedirs(savedir)

if not os.path.exists(savedir_labeled):
    os.makedirs(savedir_labeled)

label_list = []

with open('./data-crop/scale-{}.json'.format(phase), 'w+') as f_label_new:
    
    with open('./data/youtube_{}.json'.format(phase), 'r') as f_label:
        data = json.load(f_label)

        # ann = data['annotations'][ann_index]
        # images = data['images']
        # img_idxs = [im['id'] for im in images]

        anns = data['annotations']
        imgs = data['images']
        img_idxs = [im['id'] for im in imgs]

        with open('./joint2vertex.json', 'r') as f_selected:
            selectedInfo = json.loads(f_selected.readline().strip().strip('\n'))

        for i,ann in enumerate(anns):
            # if i > 100 :break
            # Crop in 4 different direction. [(-1,-1), (+1,-1), (+1,+1), (-1,+1)].
            for ii,factor in enumerate([(-1,-1), (+1,-1), (+1,+1), (-1,+1)]):
                print('>>>:{}\r'.format(i*4+ii+1), end='')
                img = imgs[img_idxs.index(ann['image_id'])]

                imagepath = os.path.join('./data/', img['name'])

                # New imagename for save to avoiding the same name.
                videoname = imagepath.split('/')[3]
                imagename = imagepath.split('/')[-1].split('.')[0] + '.jpg' 
                annid = ann['id']
                imagename = videoname + '_' + str(annid) + '_' + imagename 
                
                imagename_i = imagename.split('.jpg')[0] + '_crop_' + str(ii) + '.jpg'
                savepath = os.path.join(savedir, imagename_i)
                savepath_labeled = os.path.join(savedir_labeled, imagename_i)
                
                # Read image.
                
                image = cv2.imread(imagepath)
                # ipdb.set_trace()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgheight,imgwidth = image.shape[:2]

                averagepoints = []
                for jointid in [str(iii) for iii in range(0,21)]:
                    joint = [0,0,0]
                    jointcnt = 0
                    for pointid in selectedInfo[jointid]:
                        joint[0] += ann['vertices'][pointid][0]
                        joint[1] += ann['vertices'][pointid][1]
                        joint[2] += ann['vertices'][pointid][2]
                        jointcnt += 1
                        pass
                    joint[0] /= jointcnt
                    joint[1] /= jointcnt
                    joint[2] /= jointcnt
                    averagepoints.append(joint)

                # Write label file.
                label_dict = {}

                hand_pts = averagepoints # imgInfo["hand_pts"]
                handedness = ann['is_left'] # imgInfo["is_left"]

                hand_box_cx, hand_box_cy, hand_box_width, hand_box_height = getRecFromKeypoints(hand_pts, factor)

                crop_xmin = hand_box_cx - hand_box_width//2
                crop_xmax = hand_box_cx + hand_box_width//2
                crop_ymin = hand_box_cy - hand_box_height//2
                crop_ymax = hand_box_cy + hand_box_height//2
                
                top    = -int(crop_ymin)
                right  = -int(imgwidth - crop_xmax)
                bottom = -int(imgheight - crop_ymax)
                left   = -int(crop_xmin)

                kps = KeypointsOnImage([], shape=image.shape)
                for point in hand_pts: 
                    kps.keypoints.append(Keypoint(x=point[0], y=point[1]))

                seq = iaa.Sequential([
                    iaa.CropAndPad(px=(top,right,bottom,left), keep_size=False),# crop and pad.
                    iaa.Resize({"height": 256, "width": 256}),# resize.
                ])

                image_aug, kps_aug = seq(image=image, keypoints=kps)

                # # image with keypoints before/after augmentation (shown below)
                # image_before = kps.draw_on_image(image, size=7)
                image_after = kps_aug.draw_on_image(image_aug, size=7)
                image_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2BGR)
                cv2.imwrite(savepath_labeled, image_after)

                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
                cv2.imwrite(savepath, image_aug)
                
                label_dict["imgName"] = imagename_i
                label_dict["hand_pts"] = []
                for iii in range(len(kps_aug.keypoints)):
                    after = kps_aug.keypoints[iii]
                    label_dict["hand_pts"].append([after.x, after.y])                

                label_dict["handedness"] = handedness
                label_dict["handflag"] = 1

                label_list.append(label_dict)

                json_str = json.dumps(label_dict, cls=MyEncoder)
                f_label_new.write(json_str + '\n')

        print()

