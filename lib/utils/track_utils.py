import torch, yaml, cv2
import numpy as np
import random

# ---------------------------------
# Functions for tracking tools
# ---------------------------------
def load_yaml(path, subset=True):
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)

    if subset:
        hp = yaml_obj['TEST']
    else:
        hp = yaml_obj

    return hp


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans,
                           target_sz=None, out_mode='torch', need_bbox=False, vis=False):
    """
    SiamFC type cropping the search area online
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # For return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    if target_sz is not None:
        target_xmin = round(pos[0] - target_sz[0] / 2)
        target_xmax = round(pos[0] + target_sz[0] / 2)
        target_ymin = round(pos[1] - target_sz[1] / 2)
        target_ymax = round(pos[1] + target_sz[1] / 2)
        crop_info['original_image_bbox'] = [target_xmin, target_ymin, target_xmax, target_ymax]

    if target_sz is not None and need_bbox:

        # Now let us calculate template bbox
        patch_sz = im_patch_original.shape[0]

        x_slope = patch_sz / (context_xmax - context_xmin)
        y_slope = patch_sz / (context_ymax - context_ymin)

        target_xmin_after = left_pad - 1 + x_slope * (target_xmin - context_xmin)
        target_xmax_after = left_pad - 1 + x_slope * (target_xmax - context_xmin)
        target_ymin_after = top_pad - 1 + y_slope * (target_ymin - context_ymin)
        target_ymax_after = top_pad - 1 + y_slope * (target_ymax - context_ymin)
        scale_resize = im_patch.shape[0] / patch_sz
        crop_info['template_bbox'] = [scale_resize * target_xmin_after,
                                      scale_resize * target_ymin_after,
                                      scale_resize * target_xmax_after,
                                      scale_resize * target_ymax_after]

        if vis:
            name = "{:06d}".format(random.randint(0, 999999))
            _draw(im_patch, crop_info['template_bbox'], "./var/loader/" + name + "_track.jpg")
            _draw(im, crop_info['original_image_bbox'], "./var/loader/" + name + "_search.jpg")

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info

def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)

def _draw(image, box, name):
    """
    draw image for debugging
    """
    draw_image = np.array(image.copy())
    if box is not None:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
                    (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)
    cv2.imwrite(name, draw_image)
