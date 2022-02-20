from typing import List, Dict, Tuple, Optional
from wscreenshot import wscreenshot
import cv2
import time
import os
import numpy as np
import parse_cards

Card = str          # something like 'cspade bJoker'
Who = int
LackLevel = int     # 4, 3, 2 or 1
Image = np.ndarray

NUM_PLAYERS = 4
MAX_JIFUPAI = 4

CARD_COLORS = ['cspade', 'cheart', 'cclub', 'cdiamond']
CARD_PIONTS = ['bJoker', 'bA', 'bK', 'bQ', 'bJ', 'b10', 'b9', 'b8', 'b7', 'b6', 'b5', 'b4', 'b3', 'b2']

CARDS = [a + ' ' + b for a in CARD_COLORS for b in CARD_PIONTS[1:]]
CARDS.insert(0, 'cheart bJoker')
CARDS.insert(1, 'cspade bJoker')

CARD_TYPES = ['zhu', 'spade', 'heart', 'club', 'diamond']
# CARD_TYPES = ['zhu'] + COLORS
# ZHU_COLORS = ['nt'] + COLORS

WINTEXT = '升级角色版'

def card_to_str(card: Card):
    color, point = card.split()
    color_dict = {
        'cspade': '♠',
        'cheart': '♥',
        'cclub': '♣',
        'cdiamond': '♦'
    }
    point_dict = {
        'bJoker': 'j',
        'bA': 'A',
        'bK': 'K',
        'bQ': 'Q',
        'bJ': 'J',
        'b10': 't',
        'b9': '9',
        'b8': '8',
        'b7': '7',
        'b6': '6',
        'b5': '5',
        'b4': '4',
        'b3': '3',
        'b2': '2'
    }
    return color_dict[color]+point_dict[point]


def cards_to_str(cards: List[Card]):
    return ' '.join([card_to_str(card) for card in cards])


class JiPaiQi():
    class FrameBasicInfo:
        def __init__(self,
                     timestamp: str,
                     img: Image,
                     img_gray: Image,
                     clock_who: Optional[Who]=None,
                     is_start_clock: bool=False,
                     is_shuaipai_toast: bool=False,
                     is_game_end=False):
            self.clock_who = clock_who
            self.is_start_clock = is_start_clock
            self.is_shuaipai_toast = is_shuaipai_toast
            self.timestamp = timestamp
            self.img = img
            self.img_gray = img_gray
            self.is_game_end=is_game_end

        def __str__(self):
            return 'timestamp:%s clock_who:%s is_start_clock:%d is_shuaipai_toast%d is_game_end:%d' % (
                self.timestamp, self.clock_who, self.is_start_clock, self.is_shuaipai_toast, self.is_game_end)

    class DetailedInfo:
        def __init__(self):
            # self.output_cards: List[List[Optional[Tuple[Who, Card]]]] = []
            initial_lack_list = {k: 5 for k in CARDS}
            self.lack_list: List[Dict[Card, LackLevel]] = [initial_lack_list.copy() for _ in range(4)]

    class ShuaipaiInfo:
        def __init__(self, timestamp, who, cards):
            self.timestamp = timestamp
            self.who = who
            self.cards = cards

        def __str__(self):
            return 'timestamp:%s who:%d cards%s' % (self.timestamp, self.who, str(self.cards))

    class WsImageGenerator():
        def __init__(self):
            self.init_ws()

        def init_ws(self):
            printed_log = False
            while True:
                try:
                    self.ws = wscreenshot.Screenshot(WINTEXT)
                except Exception:
                    if not printed_log:
                        print('WsImageGenerator: ws waiting.')
                        printed_log = True
                    time.sleep(1)
                    continue
                print('WsImageGenerator: ws created.')
                break

        def next(self):
            timestamp = str(time.time())
            try:
                img = self.ws.screenshot()
            except Exception as e:
                print(e)
                self.init_ws()
                img = self.ws.screenshot()
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return timestamp, img, img_gray

    class DirImageGenerator():
        def __init__(self, img_dir):
            self.img_dir = img_dir
            img_files = os.listdir(self.img_dir)
            self.img_files = sorted(img_files)
            self.index = 0

        def next(self):
            if self.index >= len(self.img_files):
                return None

            fname = os.path.join(self.img_dir, self.img_files[self.index])
            img = cv2.imread(fname)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            self.index += 1
            # self.index %= len(self.img_files)
            return fname[:-4], img, img_gray

    def __init__(self, img_dir=None, do_record=True, loop=True):
        if img_dir:
            self.img_generator = JiPaiQi.DirImageGenerator(img_dir)
        else:
            self.img_generator = JiPaiQi.WsImageGenerator()

        self.img_dir = img_dir
        self.do_record = do_record
        self.loop = loop

        self.timestamp = None
        self.img = None
        self.img_gray = None
        self.init()
        self.initialized = False

    def init(self):
        self.my_cards: List[Card] = []
        self.out_cards: List[List[Optional[Tuple[Who, Card]]]] = []
        self.zhu_point = None
        self.zhu_color = None
        self.prev_frame_basic_info: JiPaiQi.FrameBasicInfo = None
        self.lack_list: List[List[int]] = [[MAX_JIFUPAI + 1 for _ in range(len(CARD_TYPES))] for _ in range(4)]
        self.shuaipai_info: JiPaiQi.ShuaipaiInfo = None
        self.start_timestamp = str(int(time.time()))
        # if self.img_dir is None and self.do_record:
        if self.do_record:
            os.makedirs(self.start_timestamp, exist_ok=True)

    def record_screenshot(self, timestamp, img, overwrite=False):
        if not self.do_record:
            return
        # if self.img_dir is None and self.do_record:
        timestamp = timestamp.split('\\')[-1]

        if overwrite:
            int_timestamp = str(int(float(timestamp)))
            img_fname = os.path.join(self.start_timestamp, int_timestamp + '.bmp')
            if os.path.exists(img_fname):
                os.remove(img_fname)
            cv2.imwrite(img_fname, img)
        else:
            img_fname = os.path.join(self.start_timestamp, timestamp + '.bmp')
            cv2.imwrite(img_fname, img)

    def step(self):
        frame = self.img_generator.next()
        if frame is None:
            return None

        timestamp, img, img_gray = frame
        self.timestamp, self.img, self.img_gray = timestamp, img, img_gray
        self.record_screenshot(timestamp, img, True)
        # print("step begins, timestamp=", self.timestamp)

        frame_basic_info: JiPaiQi.FrameBasicInfo = self.parse_frame_basic_info(timestamp, img, img_gray)
        if not self.initialized:
            if frame_basic_info.is_game_end:
                return False

            if not frame_basic_info.is_start_clock:
                return False
            else:
                print(timestamp, 'reinitializing')
                self.init()
                self.process_init_frame(img, img_gray)
                self.initialized = True
                self.prev_frame_basic_info = frame_basic_info
                self.log()
                self.record_screenshot(timestamp, img, False)
                return True

        if not self.prev_frame_basic_info.is_game_end and frame_basic_info.is_game_end:
            self.initialized = False
            print(timestamp, 'game end')
            self.log()
            self.record_screenshot(timestamp, img, False)
            if not self.loop:
                return None
            return True
        elif self.prev_frame_basic_info.is_game_end and frame_basic_info.is_game_end:
            return False

        # process shuaipai
        if (self.shuaipai_info is None
                and self.prev_frame_basic_info is not None
                and not self.prev_frame_basic_info.is_shuaipai_toast
                and frame_basic_info.is_shuaipai_toast):
            assert len(self.out_cards) == 0 or len(self.out_cards[-1]) in [0, 4, 1]
            need_withdraw = len(self.out_cards) != 0 and len(self.out_cards[-1]) not in [0,4]
            if not need_withdraw:
                who_shuaipai = self.prev_frame_basic_info.clock_who
                shuaipai_cards = parse_cards.parse_out_cards(img, img_gray, who_shuaipai)
            else:
                who_shuaipai = self.out_cards[-1][0][0]
                shuaipai_cards = self.out_cards[-1][0][1]
                self.withdraw_cards()
            self.shuaipai_info = JiPaiQi.ShuaipaiInfo(
                timestamp, who_shuaipai, shuaipai_cards
            )
            self.prev_frame_basic_info.clock_who = who_shuaipai
            self.log()
            print(timestamp, 'start shuaipai, who=%d, cards=%s, withdraw=%d' % (
                self.shuaipai_info.who, str(self.shuaipai_info.cards), need_withdraw))
            self.record_screenshot(timestamp, img, False)
            return False

        if self.shuaipai_info is not None:
            shuaipai_cards = parse_cards.parse_out_cards(img, img_gray, self.shuaipai_info.who)
            if shuaipai_cards == self.shuaipai_info.cards:
                self.log()
                print(timestamp, 'waiting for shuaipai, who=%d, cards=%s' % (
                    self.shuaipai_info.who, str(self.shuaipai_info.cards)))
                self.record_screenshot(timestamp, img, False)
                time.sleep(0.1)
                return False
            else:
                assert len(shuaipai_cards) >= 1
                self.log()
                print(timestamp, 'end shuaipai, who=%d, old_cards=%s, new_cards=%s' %
                      (self.shuaipai_info.who, str(self.shuaipai_info.cards), str(shuaipai_cards)))
                self.record_screenshot(timestamp, img, False)
                self.shuaipai_info = None

        who, prev_who = frame_basic_info.clock_who, self.prev_frame_basic_info.clock_who
        if who is None or prev_who is None:
            return False

        if who != prev_who:
            # put cards
            if len(self.out_cards) == 0 or len(self.out_cards[-1]) == 4:
                self.out_cards.append([])

            # need_fill = 4 - len(self.out_cards[-1])
            who_out = prev_who
            while True:
                if len(self.out_cards[-1]) == 4:
                    break
                if who_out == who:
                    break
                out_cards = parse_cards.parse_out_cards(img, img_gray, who_out, self.my_cards, self.out_cards)
                self.put_cards(timestamp, who_out, out_cards)
                who_out = (who_out + 1) % 4

            self.log()
            self.prev_frame_basic_info = frame_basic_info
            self.record_screenshot(timestamp, img, False)
            return True
        else:
            cards_who = parse_cards.parse_out_cards(img, img_gray, who, self.my_cards, self.out_cards)
            if len(cards_who) == 0:
                return False

            prev_cards_who = parse_cards.parse_out_cards(
                self.prev_frame_basic_info.img,
                self.prev_frame_basic_info.img_gray,
                self.prev_frame_basic_info.clock_who,
                self.my_cards, self.out_cards)
            if not (prev_cards_who == cards_who or len(prev_cards_who) == 0):
                assert len(parse_cards.parse_self_cards(img, img_gray)) == 0
                return False
            if prev_cards_who == cards_who:
                return False
            assert len(prev_cards_who) == 0
            assert len(self.out_cards[-1]) >= 1
            # need_fill = 4 - len(self.out_cards[-1])
            # for i in range(need_fill - 1, -1, -1):
            #     who_out = (who - i) % 4
            #     out_cards = parse_cards.parse_out_cards(img, img_gray, who_out)
            #     self.put_cards(timestamp, who_out, out_cards)
            who_out = (self.out_cards[-1][-1][0] + 1) % 4
            while len(self.out_cards[-1]) < 4:
                out_cards = parse_cards.parse_out_cards(img, img_gray, who_out, self.my_cards, self.out_cards)
                self.put_cards(timestamp, who_out, out_cards)
                who_out = (who + 1) % 4

            self.log()
            self.prev_frame_basic_info = frame_basic_info
            self.record_screenshot(timestamp, img, False)
            return True

    def is_zhu(self, card):
        color, point = card.split()
        color = color[1:]
        point = point[1:]
        if point == 'Joker':
            return True
        if point == self.zhu_point:
            return True
        if self.zhu_color != 'nt' and color == self.zhu_color:
            return True
        return False

    def get_card_type(self, card):
        # zhu spade heart club diamond
        # z s h c d
        if self.is_zhu(card):
            return 'zhu'

        color, point = card.split()
        color = color[1:]
        return color

    def analyze_out_cards(self, cards):
        # num_triples, num_pairs, num_singles, num_others
        round_type = self.get_card_type(self.out_cards[-1][0][1][0])

        cards_with_right_type = [x for x in cards if self.get_card_type(x) == round_type]


        # points = [x.split()[1][1:] for x in cards_with_right_type]

        from collections import defaultdict
        stat = defaultdict(int)
        for card in cards_with_right_type:
            stat[card] += 1
        values = list(stat.values())
        assert all(1 <= x <= self.jifupai for x in values)
        # s = len(cards_with_right_type)
        # p = sum(x // 2 for x in values)
        # t = sum(x // 3 for x in values)
        # q = sum(x // 4 for x in values)
        stat_by_type = [0] * 4
        for x in values:
            stat_by_type[x] += 1
        return stat_by_type

    def withdraw_cards(self):
        assert len(self.out_cards[-1]) == 1
        who, cards = self.out_cards[-1][0]
        print(self.timestamp, 'withdraw_cards, who=%d, cards=%s' % (who, cards_to_str(cards)))
        self.out_cards.pop()

    def update_lack(self, who, out_card):
        assert len(self.out_cards[-1]) != 0
        if len(self.out_cards[-1]) == 1:
            return

        first_card = self.out_cards[-1][0][1]
        card_type = self.get_card_type(first_card[0])

        s, p, t, q = self.analyze_out_cards(first_card)
        s1, p1, t1, q1 = self.analyze_out_cards(out_card)

        def get_lack_level(s, p, t, q, s1, p1, t1, q1):
            lack_single = (s1 + p1 * 2 + t1 * 3 + q1 * 4 < s + p * 2 + t * 3 + q * 4)
            # lack_pair = (p1 < p)
            lack_triple = (t1 + q1 < t + q)
            lack_quadra = (q1 < q)

            p0 = min(p, p1)
            p -= p0
            p1 -= p0
            t0 = min(t, t1)
            t -= t0
            t1 -= t0
            q0 = min(q, q1)
            q -= q0
            q1 -= q0
            # if q != 0:
            #     t1 -= min(q, t1)
            #     q -= min(q, t1)
            # if
            if q != 0 and t != 0:
                pass
            elif q != 0 and t == 0:
                q0 = min(q, t1)
                t1 -= q0
                q -= q0
            elif q == 0 and t != 0:
                t0 = min(t, q1)
                t -= t0
                q1 -= t0
            else:
                pass
            lack_pair = (p1 + t1 + q1 * 2) < (p + t + q * 2)

            if lack_single:
                lack_level = 1
            elif lack_pair:
                lack_level = 2
            elif lack_triple:
                lack_level = 3
            elif lack_quadra:
                lack_level = 4
            else:
                lack_level = 5
            return lack_level

        lack_level = get_lack_level(s, p, t, q, s1, p1, t1, q1)
        type_idx = CARD_TYPES.index(card_type)

        if lack_level < self.lack_list[who][type_idx]:
            self.lack_list[who][type_idx] = lack_level
            print('lack:', who, self.lack_list[who])

    def analyze(self):
        assert len(self.out_cards[-1]) != 0
        if len(self.out_cards[-1]) == 1:
            return
        who, out_card = self.out_cards[-1][-1]
        self.update_lack(who, out_card)

    def put_cards(self, timestamp, who, cards):
        assert len(cards) != 0
        if len(self.out_cards[-1]) != 0:
            assert (self.out_cards[-1][-1][0] + 1) % 4 == who
            assert len(self.out_cards[-1][-1][1]) == len(cards)
        self.out_cards[-1].append((who, cards))
        assert len(self.out_cards[-1]) <= 4
        print(timestamp, 'put_cards, who=%d, cards=%s' % (who, cards_to_str(cards)))
        self.analyze()

    def parse_frame_basic_info(self, timestamp, img, img_gray):
        is_game_end = parse_cards.parse_game_end(img_gray)
        clock_info = parse_cards.parse_clock(img_gray)
        if clock_info is None:
            return JiPaiQi.FrameBasicInfo(
                timestamp=timestamp,
                img=img,
                img_gray=img_gray,
                is_game_end=is_game_end)
        clock_who, is_start_clock = clock_info
        is_shuaipai_toast = parse_cards.parse_shuaipai(img_gray)
        res = JiPaiQi.FrameBasicInfo(
            timestamp=timestamp,
            img=img,
            img_gray=img_gray,
            clock_who=clock_who,
            is_start_clock=is_start_clock,
            is_shuaipai_toast=is_shuaipai_toast,
            is_game_end=is_game_end)
        return res

    def process_init_frame(self, img, img_gray):
        self.my_cards = parse_cards.parse_self_cards(img, img_gray)
        card_nums = [25, 33, 39, 45, 52, 60]
        assert len(self.my_cards) in card_nums
        self.jifupai = 2 + card_nums.index(len(self.my_cards)) // 2
        zhu_point = parse_cards.parse_zhu_point(img_gray)
        self.zhu_point = zhu_point[1:]
        if self.zhu_point == '1':
            self.zhu_point = '10'
        self.zhu_color = parse_cards.parse_zhu_color(img_gray)[1:]

    def log_image(self):
        pass

    def log(self):
        s = (
            'initialized: %s' % self.initialized +
            ' timestamp: %s' % self.timestamp +
            ' zhu_point: %s' % self.zhu_point +
            ' zhu_color: %s' % self.zhu_color +
            ' jifupai: %s' % self.jifupai +
            ' my_cards: %s' % self.my_cards +
            ' out_cards: %s' % self.out_cards +
            ' shuaipai_info %s' % self.shuaipai_info +
            ' lack_list: %s' % self.lack_list
        )
        replace_dict = {
            'cspade': '♠',
            'cheart': '♥',
            'cclub': '♣',
            'cdiamond': '♦',
            'bJoker': 'j',
            'bA': 'A',
            'bK': 'K',
            'bQ': 'Q',
            'bJ': 'J',
            'b10': 't',
            'b9': '9',
            'b8': '8',
            'b7': '7',
            'b6': '6',
            'b5': '5',
            'b4': '4',
            'b3': '3',
            'b2': '2'
        }
        for k in replace_dict:
            s = s.replace(k, replace_dict[k])

        print(s)


def jipaiqi_to_table_content(jipaiqi):
    COLOR_DICT = {
        'cspade': '♠',
        'cheart': '♥',
        'cclub': '♣',
        'cdiamond': '♦'
    }

    POINT_DICT = {
        'bJoker': '王',
        'bA': 'A',
        'bK': 'K',
        'bQ': 'Q',
        'bJ': 'J',
        'b10': '⒑',
        'b9': '9',
        'b8': '8',
        'b7': '7',
        'b6': '6',
        'b5': '5',
        'b4': '4',
        'b3': '3',
        'b2': '2'
    }

    if not jipaiqi.initialized:
        return None

    # s = (
    #         'initialized: %s' % self.initialized +
    #         ' timestamp: %s' % self.timestamp +
    #         ' zhu_point: %s' % self.zhu_point +
    #         ' zhu_color: %s' % self.zhu_color +
    #         ' jifupai: %s' % self.jifupai +
    #         ' my_cards: %s' % self.my_cards +
    #         ' out_cards: %s' % self.out_cards +
    #         ' shuaipai_info %s' % self.shuaipai_info +
    #         ' lack_list: %s' % self.lack_list
    # )

    # 4 * 5 * 15
    PLAYERS = 4
    ROWS = 5
    COLS = 15
    tables = [[[(' ', 'white', 'black') for _ in range(COLS)] for _ in range(ROWS)] for _ in range(PLAYERS)]

    row = 0
    for player in range(PLAYERS):
        for col in range(1, COLS):
            text = POINT_DICT[CARD_PIONTS[col - 1]]
            tables[player][row][col] = text, 'white', 'black'

    col = 0
    for player in range(PLAYERS):
        for row in range(1, ROWS):
            text = COLOR_DICT[CARD_COLORS[row - 1]]
            tables[player][row][col] = text, 'white', 'red' if row % 2 == 0 else 'black'

    out_cards_by_player = [[] for _ in range(PLAYERS)]
    for round in jipaiqi.out_cards:
        for player, out_cards in round:
            out_cards_by_player[player].extend(out_cards)

    out_card_dict_by_player = [{card: 0 for card in CARDS} for _ in range(PLAYERS)]
    for player in range(PLAYERS):
        for card in out_cards_by_player[player]:
            out_card_dict_by_player[player][card] += 1

    remain_card_nums = {card: jipaiqi.jifupai for card in CARDS}
    for card in jipaiqi.my_cards:
        remain_card_nums[card] -= 1
    for player in range(PLAYERS):
        if player == 2:
            continue
        for card, card_num in out_card_dict_by_player[player].items():
            remain_card_nums[card] -= card_num

    for player in range(PLAYERS):
        for row in range(1, ROWS):
            color = CARD_COLORS[row - 1]
            for col in range(1, COLS):
                point = CARD_PIONTS[col - 1]
                card = color + ' ' + point
                if card in ['cclub bJoker', 'cdiamond bJoker']:
                    continue
                if player != 2:
                    num = out_card_dict_by_player[player][card]
                else:
                    num = remain_card_nums[card]
                text = str(num)
                tables[player][row][col] = text, 'gold' if jipaiqi.is_zhu(card) else 'white', 'gray' if num == 0 else 'black'

    details = []
    CARD_TYPE_STRS = ['主', '♠', '♥', '♣', '♦']
    for player in range(PLAYERS):
        s = '玩家:%d' % player
        for card_type in range(len(CARD_TYPE_STRS)):
            lack_level = jipaiqi.lack_list[player][card_type]
            if lack_level >= 4:
                continue

            s += ' 缺' + CARD_TYPE_STRS[card_type] * lack_level
        details.append(s)

    return tables, details


if __name__ == '__main__':
    jipaiqi = JiPaiQi(img_dir='bug_1645289726', do_record=False, loop=False)
    while True:
        res = jipaiqi.step()
        if res is None:
            break
        if res:
            print('table_content:')
            tables, details = jipaiqi_to_table_content(jipaiqi)
            for player in range(4):
                print('=== player', player)
                for row in range(5):
                    for col in range(15):
                        text, fg, bg = tables[player][row][col]
                        print(text, end=' ')
                    print()
                print()
            print()


