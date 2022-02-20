import os
import cv2 as cv
import numpy as np
from non_max_suppression import non_max_suppression


big_char_template_files = [x for x in os.listdir('templates') if x[0] == 'b']
big_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                      for template_file in big_char_template_files]

fallback_char_template_files = [x for x in os.listdir('templates') if x[0] == 'f' and x != 'fJoker']
fallback_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                      for template_file in fallback_char_template_files]

black_char_template_files = ['cspade.bmp', 'cclub.bmp']
black_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                             for template_file in black_char_template_files]

color_char_template_files = ['cspade.bmp', 'cclub.bmp', 'cheart.bmp', 'cdiamond.bmp', 'cjoker.bmp']
color_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                           for template_file in color_char_template_files]

small_char_template_files = [x for x in os.listdir('templates') if x[0] == 's']
small_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                      for template_file in small_char_template_files]

zhucolor_char_template_files = [x for x in os.listdir('templates') if x[0] == 'z']
zhucolor_char_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                           for template_file in zhucolor_char_template_files]

gamestart_templates_files = ['mgamestart.bmp']
gamestart_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                       for template_file in gamestart_templates_files]

gameend_templates_files = ['mgameend.bmp']
gameend_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                       for template_file in gameend_templates_files]

shuaipai_templates_files = ['mshuaipai.bmp']
shuaipai_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                      for template_file in shuaipai_templates_files]

clock_templates_files = ['mclock.bmp', 'mclockstart.bmp']
clock_templates = [(template_file.split('.')[0], cv.imread('templates/' + template_file, 0))
                   for template_file in clock_templates_files]


def parse_cards(img_gray, templates, region=None, threshold = 0.85):
    rects = []
    probs = []
    labels = []

    if region is None:
        x1, y1, x2, y2 = 0, 0, img_gray.shape[1], img_gray.shape[0]
    else:
        x1, y1, x2, y2 = region
    img_gray = img_gray[y1:y2, x1: x2]

    for (label, template) in templates:

        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)
        w, h = template.shape[::-1]
        for loc1 in zip(*loc[::-1]):
            rects.append([loc1[0], loc1[1], loc1[0] + w, loc1[1] + h])
            probs.append(res[loc1[1], loc1[0]])
            labels.append(label)

    picks = non_max_suppression(np.array(rects), np.array(probs), overlapThresh=0.1)
    for rect in rects:
        rect[0] += x1
        rect[1] += y1
        rect[2] += x1
        rect[3] += y1
    result = [[rects[j], probs[j], labels[j]] for j in picks]
    result.sort(key=lambda x: x[0][0])
    return result


def parse_cards_with_color(img_rgb, img_gray, region):
    result = []

    res1 = parse_cards(img_gray, color_char_templates, region)
    if len(res1) == 0:
        return []

    for (x1, y1, x2, y2), prob, label in res1:
        x, y = x1 - 3, y1 - 22 - 3
        w, h = 28, 27
        res2 = parse_cards(img_gray, big_char_templates, [x, y, x + w, y + h], threshold=0.65)
        if len(res2) != 1:
            return "OUT_CARDS_BLOCKED"
        (x1_, y1_, x2_, y2_), prob_, label_ = res2[0]
        if label_ == 'bJoker':
            b, g, r = img_rgb[y1_ + 4, x1_ + 10]
            if b < 50 and g < 50 and r > 100:
                label = 'cheart'
            else:
                label = 'cspade'
        rect = [min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_)]
        result.append([rect, prob * prob_, label + ' ' + label_])
        if prob * prob_ < 0.9:
            print("warning: low prob, ", label + ' ' + label_)
    return result


def visualize(img, result):
    img_new = img.copy()
    for ((x1, y1, x2, y2), prob, label) in result:
        cv.rectangle(img_new, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(img_new, label.split()[-1][1], (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0, 0, 0))

    cv.imwrite('res.png', img_new)


def parse_self_cards(img_rgb, img_gray):
    region = [88, 771, 1656, 837]
    result = parse_cards_with_color(img_rgb, img_gray, region)
    return sorted(x[2] for x in result)


def _parse_out_cards(img_rgb, img_gray, who):
    if who == 0:
        region = [805-250, 315, 805+250, 359]
    elif who == 1:
        region = [170, 429, 791, 476]
    elif who == 3:
        region = [800, 429, 1440, 476]
    else:
        region = [805-250, 561, 805+250, 605]
    result = parse_cards_with_color(img_rgb, img_gray, region)
    return result


def _parse_self_out_cards_fallback(img_rgb, img_gray, my_cards, all_out_cards):
    def diff(cards1, cards2):
        for card in cards2:
            cards1.remove(card)
        return cards1

    my_history_out_cards = [card for round in all_out_cards for (who, cards) in round if who == 2 for card in cards]
    remain_cards = diff(my_cards, my_history_out_cards)
    hand_remain_cards = parse_self_cards(img_rgb, img_gray)
    out_cards = diff(remain_cards, hand_remain_cards)
    return sorted(out_cards)

def _parse_player3_out_cards_fallback(img_rgb, img_gray):
    region1 = [946, 429, 1440, 476]
    result1 = parse_cards_with_color(img_rgb, img_gray, region1)

    region2 = [800, 429, 944, 476]
    def apply_alpha(color_char_templates):
        alpha = np.array([0.57692308, 0.58490566, 0.57692308])
        new = np.array([226.13333333, 250.83870968, 212.26666667])
        alpha = np.mean(alpha)
        new = np.mean(new)
        new_color_char_templates = color_char_templates.copy()
        for x in new_color_char_templates:
            x[1][:] = (alpha * new + (1 - alpha) * x[1]).astype(np.int16)
        return new_color_char_templates

    res1 = parse_cards(img_gray, apply_alpha(color_char_templates), region2)
    if len(res1) == 0:
        return []

    result2 = []
    for (x1, y1, x2, y2), prob, label in res1:
        if label == 'cjoker':
            b, g, r = img_rgb[y1 + 0, x1 + 4]
            if b < 50 and g < 50 and r > 100:
                label = 'cheart'
            else:
                label = 'cspade'
            label_ = 'bJoker'
            prob_ = prob
            x1_, y1_, x2_, y2_ = x1, y1, x2, y2
        else:
            x, y = x1 - 3, y1 - 22
            w, h = 28, 10
            res2 = parse_cards(img_gray, fallback_char_templates, [x, y, x + w, y + h])
            assert len(res2) != 0
            (x1_, y1_, x2_, y2_), prob_, label_ = res2[0]
            assert label_[0] == 'f'
            label_ = 'b' + label_[1:]
        rect = [min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_)]
        result2.append([rect, prob * prob_, label + ' ' + label_])
        if prob * prob_ < 0.9:
            print("warning: low prob, ", label + ' ' + label_)
    return result1 + result2


class OutCardsBlockedException(Exception):
    def __init__(self, who):
        super(OutCardsBlockedException, self).__init__()
        self.who = who

def parse_out_cards(img_rgb, img_gray, who, my_cards=None, all_out_cards=None):
    result = _parse_out_cards(img_rgb, img_gray, who)
    if result == "OUT_CARDS_BLOCKED":
        # assert who in [2, 3]
        if who not in [2, 3]:
            raise OutCardsBlockedException(who)
        if who == 2:
            result = _parse_self_out_cards_fallback(img_rgb, img_gray, my_cards, all_out_cards)
            return result
        elif who == 3:
            result = _parse_player3_out_cards_fallback(img_rgb, img_gray)
            return sorted([x[2] for x in result])
    return sorted([x[2] for x in result])


def parse_clock(img_gray):
    regions = [
        [930, 156, 980, 206],
        [166, 395, 216, 443],
        [166, 713, 216, 765],
        [1446, 395, 1496, 443],
    ]
    points = [
        [5,1], [2,5], [8,5], [5, 9], [2, 12], [8, 12], [5, 15],
        [18, 1], [15,5],[21,5],[18,8],[15,12],[21,12], [18, 15]
    ]
    colors = [
        'black', 'white', 'black', 'black', 'white', 'black', 'black',
        'black', 'black', 'black', 'black', 'white', 'black', 'black',
    ]
    who = []
    start = None
    for i, region in enumerate(regions):
        result = parse_cards(img_gray, [clock_templates[0]], region)
        is_start = False
        if result:
            result2 = parse_cards(img_gray, [clock_templates[1]], region)
            if result2:
                x, y, x1, y1 = result2[0][0]
                def is_right_color(point, color):
                    if color == 'white':
                        return img_gray[y + point[1], x + point[0]] >= 192
                    elif color == 'black':
                        return img_gray[y + point[1], x + point[0]] < 64
                is_39 = all(is_right_color(point, color) for (point, color) in zip(points, colors))
                is_start = is_39

        if result:
            who.append(i)
        start = start or is_start

    if len(who) != 1:
        return None

    return who[0], start


def _parse_shuaipai(img_gray):
    region = [785, 458, 846, 481]
    res = parse_cards(img_gray, shuaipai_templates, region)
    return res

def parse_shuaipai(img_gray):
    res = _parse_shuaipai(img_gray)
    return bool(res)


def parse_zhu_point(img_gray):
    region1 = [73, 949, 83, 962]
    res1 = parse_cards(img_gray, small_char_templates, region1)
    return res1[0][2]

def _parse_zhu_color(img_gray):
    region1 = [103, 943, 126, 961]
    res1 = parse_cards(img_gray, zhucolor_char_templates, region1, threshold=0.7)
    return res1

def parse_zhu_color(img_gray):
    result = _parse_zhu_color(img_gray)
    return result[0][2]

def parse_game_start(img_gray):
    region = [1380, 720, 1396, 754]
    result = parse_cards(img_gray, gamestart_templates, region)
    return bool(result)

def parse_game_end(img_gray):
    region = [801, 761, 868, 792]
    result = parse_cards(img_gray, gameend_templates, region)
    return bool(result)


# def parse_game_end(img_gray):
#     regions = [
#         [493, 45, 521, 82],
#         [46, 179, 70, 213],
#         [35, 562, 64, 598],
#         [998, 181, 1024, 213]
#     ]
#     for region in regions:
#         result = parse_cards(img_gray, gameend_templates, region)
#         if result:
#             return True
#     return False


if __name__ == '__main__':

    import time
    img_rgb = cv.imread('bug_1645289726/1645289885.bmp')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    print(parse_clock(img_gray))
    t1 = time.time()
    # result = _parse_out_cards(img_rgb, img_gray, 3)
    result = _parse_player3_out_cards_fallback(img_gray, img_gray)
    t2 = time.time()
    print(t2 - t1)
    print(result)

    print(len(result), result)
    visualize(img_rgb, result)
    for x in result:
        print(x)
