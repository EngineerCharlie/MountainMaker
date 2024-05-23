#!/usr/bin/env python

# Paint program

import pygame, os, sys, time, random, math

# canvas size
RES = 256, 256

# brush size (large)
BRUSH = 21

# brush size (small)
BRUSH_SMALL = 7

# line size (large)
LSIZE = 14

# line size (small)
LSIZE_SMALL = 4

# number of points for Bézier curve
BEZSEG = 100

# airbrush density
AIRDENS = 10

# airbrush dot size
AIRSIZE = 4

# flood fill speed (lines per frame)
FILL_SPEED = 256

# palette box size
PALBW = 50

# palette columns
PALHE = 3

# toolbox pixel height
TOOL_HEIGHT = 2 * 50

# background color
COLBG = 0x202020

PALETTE_NUMBER = 0

# color palettes in RGB hex:

# default colors
# peak juhasz
cols_j = [
    0xFFFFFF,
    0xC0C0C0,
    0x808080,
    0x404040,
    0x5C4033,
    0x91624C,
    0x004084,
    0x2475CC,
    0x056232,
    0x16A45A,
    0xF56A00,
]

cols_dv = [
    0x0D1B2A,
    0x1B263B,
    0x415A77,
    0x778DA9,
    0xE0E1DD,
    0x132A13,
    0x31572C,
    0x4F772D,
    0x90A955,
    0xECF39E,
    0x562A0E,
    0x78380C,
    0xC8691C,
    0xD09259,
    0xE4CEAF,
]

cols_s = [
    0,
    0x404040,  # black, white, and grays
    0x606060,
    0x808080,
    0xA0A0A0,
    0xB0B0B0,
    0xD0D0D0,
    0xFFFFFF,
    0xFF0000,  # full-intensity colors
    0xFFB000,
    0x00FF00,
    0x0000FF,
    0xFFFF00,
    0x00FFFF,
    0xFF00FF,
    0xAA0000,  # darker colors
    0x884000,
    0x00AA00,
    0x0000AA,
    0xAAAA00,
    0x00AAAA,
    0xAA00AA,
    0xFF8080,  # pastel colors
    0xFFD080,
    0x80FF80,
    0x8080FF,
    0xFFFF80,
    0x80FFFF,
    0xFF80FF,
]

# GNOME
cols_g = [
    0,
    0xFFFFFF,
    0x555753,
    0xCC0000,
    0xF57900,
    0xEDD400,
    0x73D216,
    0x3465A4,
    0x75507B,
    0xC17D11,
]

# PICO-8
cols_p = [
    0,
    0xFFFFFF,
    0x1D2B53,
    0x7E2553,
    0x008751,
    0xAB5236,
    0x5F574F,
    0xC2C3C7,
    0xFFF1E8,
    0xFF004D,
    0xFFA300,
    0xFFEC27,
    0x00E436,
    0x29ADFF,
    0x83769C,
    0xFF77A8,
    0xFFCCAA,
]

# DeluxePaint (Dan Silva)
cols_d = [
    0,
    0xA0A0A0,
    0xEE0000,
    0xAA0000,
    0xDD8800,
    0xFFEE00,
    0x88FF00,
    0x008800,
    0x00BB66,
    0x00DDDD,
    0x00AAFF,
    0x0077CC,
    0x0000FF,
    0x7700FF,
    0xCC00EE,
    0xCC0088,
    0x662200,
    0xEE5522,
    0xAA5522,
    0xFFCCAA,
    0x333333,
    0x444444,
    0x555555,
    0x666666,
    0x777777,
    0x888888,
    0x999999,
    0xAAAAAA,
    0xCCCCCC,
    0xDDDDDD,
    0xEEEEEE,
    0xFFFFFF,
]

# MS Paint classic
cols_msp = [
    0xFFFFFF,
    0,
    0xC0C0C0,
    0x808080,
    0xFF0000,
    0x800000,
    0xFFFF00,
    0x808000,
    0x00FF00,
    0x008000,
    0x00FFFF,
    0x008080,
    0x0000FF,
    0x000080,
    0xFF00FF,
    0x800080,
    0xFFFF80,
    0x808040,
    0x00FF80,
    0x004040,
    0x80FFFF,
    0x0080FF,
    0x8080FF,
    0x004080,
    0xFF0080,
    0x400080,
    0xFF8040,
    0x804000,
]

# Pinta
cols_pt = [
    0xFFFFFF,
    0,
    0xA0A0A0,
    0x808080,
    0x404040,
    0x303030,
    0xFF0000,
    0xFF7F7F,
    0xFF6A00,
    0xFFB27F,
    0xFFD800,
    0xFFE97F,
    0xB6FF00,
    0xDAFF7F,
    0x4CFF00,
    0xA5FF7F,
    0x00FF21,
    0x7FFF8E,
    0x00FF90,
    0x7FFFC5,
    0x00FFFF,
    0x7FFFFF,
    0x0094FF,
    0x7FC9FF,
    0x0026FF,
    0x7F92FF,
    0x4800FF,
    0xA17FFF,
    0xB200FF,
    0xD67FFF,
    0xFF00DC,
    0xFF7FED,
    0xFF006E,
    0xFF7FB6,
]

# SuperPaint (Richard Shoup)
cols_sp = [
    0xE0E0E0,
    0,
    0xE00300,
    0x00E003,
    0x0300E0,
    0xE08D00,
    0xE000E0,
    0x404040,
    0x808080,
    0xC0C0C0,
    0xE0AE53,
    0x70A8E0,
    0x6FE078,
    0xE0846F,
    0xBF8500,
    0xDFDF00,
]

# Minecraft dyes (before 1.12) from https://minecraft.wiki/w/Dyeing
cols_mc1 = [
    0xFFFFFF,
    0x999999,
    0x4C4C4C,
    0x191919,
    0x664C33,
    0x993333,
    0xD87F33,
    0xE5E533,
    0x7FCC19,
    0x667F33,
    0x4C7F99,
    0x6699D8,
    0x334CB2,
    0x7F3FB2,
    0xB24CD8,
    0xF27FA5,
]

# current Minecraft dyes from https://minecraft.wiki/w/Dyeing#Color_values
cols_mc2 = [
    0xF9FFFE,
    0x9D9D97,
    0x474F52,
    0x1D1D21,
    0x835432,
    0xB02E26,
    0xF9801D,
    0xFED83D,
    0x80C71F,
    0x5E7C16,
    0x169C9C,
    0x3AB3DA,
    0x3C44AA,
    0x8932B8,
    0xC74EBD,
    0xF38BAA,
]

# Mostly empty palette (meant for picking colors from images)
cols_pick = [
    0,
    0xFFFFFF,
]

# palettes and palette names
palettes = [
    (cols_dv, "Sexy diego palette"),
    (cols_j, "Peak Juhasz"),
    (cols_s, "default"),
    (cols_g, "GNOME"),
    (cols_p, "PICO-8"),
    (cols_d, "DeluxePaint"),
    (cols_msp, "MS Paint"),
    (cols_pt, "Pinta"),
    (cols_sp, "SuperPaint"),
    (cols_mc1, "old Minecraft"),
    (cols_mc2, "new Minecraft"),
    (cols_pick, "empty"),
]

# palette rows
num_colours = len(palettes[PALETTE_NUMBER][1])
print(num_colours)
extra_row = -1 if num_colours % 3 == 0 else 1
print(num_colours % 3 == 0)
PALROWS = (num_colours // 3) + extra_row  # 4  # RES[1] // PALBW - 3

# tool names
tname = [
    "fill tool",
    "continuous freehand",
    "straight lines",
    "mountainify",
    "curves",
    "airbrush",
    "dotted freehand",
]

T_FIL = 0
T_CON = 1
T_STR = 2
T_MTN = 3
T_CUR = 3
T_AIR = 4
T_DOT = 5

T_ALL = "fil", "con", "str"

# load icon images
icons = []
icons_dark = []
for n in T_ALL:
    p = "t_%s.png" % n
    img = pygame.image.load(os.path.join("PaintApp/img", p))
    icons.append(img)
    img2 = img.copy()
    for y in range(7, 43):
        for x in range(7, 43):
            if img2.get_at((x, y))[0] > 100:
                img2.set_at((x, y), (0, 0, 0))
            else:
                img2.set_at((x, y), (170, 170, 170))
    icons_dark.append(img2)

# brico_s = pygame.image.load(os.path.join("PaintApp/img", "brsmall.png"))
# brico_b = pygame.image.load(os.path.join("PaintApp/img", "brbig.png"))
b_mtn = pygame.image.load(os.path.join("PaintApp/img", "mtn_s.png"))
b_undo = pygame.image.load(os.path.join("PaintApp/img", "undo.png"))
b_clear = pygame.image.load(os.path.join("PaintApp/img", "clear.png"))


def get_top(ca, orig, ys):
    for y in range(ys - 1, -1, -1):
        if ca[y] != orig:
            return y + 1
    return 0


def get_bot(ca, orig, ys):
    for y in range(ys + 1, len(ca)):
        if ca[y] != orig:
            return y - 1
    return len(ca) - 1


# list of flood fill regions left to process
ra = []


def fill2(surf, p, col):
    "Start flood fill at pos p"
    global ra

    orig = surf.get_at(p)
    # check if target pixel already has the chosen color
    if col == 256**2 * orig[0] + 256 * orig[1] + orig[2]:
        return

    ca = [surf.get_at((p[0], y)) for y in range(surf.get_height())]
    top, bot = get_top(ca, orig, p[1]), get_bot(ca, orig, p[1])
    ra.append((p[0], top, bot, 1, orig, col))
    ra.append((p[0], top, bot, -1, orig, col))
    pygame.draw.line(surf, col, (p[0], top), (p[0], bot))


def do_fill(surf):
    "Perform a flood fill step"
    if not ra:
        return
    x, top, bot, di, orig, col = ra.pop(0)
    x += di
    if x < 0 or x >= surf.get_width():
        return
    cr = [surf.get_at((x, y)) for y in range(surf.get_height())]
    lastbot = -1

    for y in range(top, bot + 1):
        if cr[y] != orig or y <= lastbot:
            continue
        top2, bot2 = get_top(cr, orig, y), get_bot(cr, orig, y)
        if bot2 == lastbot:
            continue
        lastbot = bot2
        ra.append((x, top2, bot2, di, orig, col))
        pygame.draw.line(surf, col, (x, top2), (x, bot2))
        ra.append((x, top2, bot2, -di, orig, col))


def bezier(surf, col, br, pos):
    "Draw a quadratic Bézier curve to surface"
    a, c, b = pos
    poi = []
    for tt in range(0, BEZSEG + 1):
        t = tt / BEZSEG
        # http://blog.pkh.me/p/33-deconstructing-be%CC%81zier-curves.html
        xp = (1 - t) ** 2 * a[0] + 2 * (1 - t) * t * b[0] + t**2 * c[0]
        yp = (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * b[1] + t**2 * c[1]
        poi.append((xp, yp))
    for n in range(len(poi) - 1):
        pygame.draw.line(surf, col, poi[n], poi[n + 1], width=br)


def is_act(a, b):
    "Is this the active tool?"
    if a == b:
        return icons_dark[a]
    else:
        return icons[a]


class Paint:
    def __init__(self):
        pygame.init()
        self.palnum = PALETTE_NUMBER
        height = max(RES[1], TOOL_HEIGHT + PALBW * PALROWS)
        self.screen = pygame.display.set_mode((RES[0] + PALHE * PALBW, height))
        self.clock = pygame.time.Clock()
        # if len(sys.argv) > 1:
        #     self.img = pygame.image.load(sys.argv[1]).convert()
        #     w, h = self.img.get_size()
        #     if w > RES[0] or h > RES[1]:
        #         asp_can = RES[0] / RES[1]
        #         asp_img = w / h

        #         # scale to fit canvas
        #         if asp_img > asp_can:
        #             w = RES[0]
        #             h = RES[0] / asp_img
        #         else:
        #             w = RES[1] * asp_img
        #             h = RES[1]
        #         self.img = pygame.transform.smoothscale(self.img, (w, h))
        # else:
        self.img = pygame.Surface(RES)
        self.img.fill(0xFFFFFF)
        self.mdown = False
        self.cols = palettes[self.palnum][0]
        self.col = 0
        self.colpic = pygame.Surface((PALHE * PALBW, height))
        self.getcolpic()
        self.undo = [self.img.copy()]
        self.tool = T_CON
        self.small_brush = True
        self.line_start = 0, 0
        self.bezier = []
        self.hide = False
        self.title()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.undo.append(self.img.copy())
                self.img.fill(0xFFFFFF)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                pygame.image.save(self.img, time.strftime("%y%m%d_%H%M%S.png"))
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and pygame.mouse.get_pos()[0] < RES[0]:
                    self.mdown = True
                    self.lastpos = pygame.mouse.get_pos()
                    if self.tool == T_STR:  # straight lines
                        self.line_start = pygame.mouse.get_pos()
                    if self.tool == T_CUR and len(self.bezier) == 0:  # curves
                        self.bezier = [pygame.mouse.get_pos()]
                    if self.tool == T_CUR and len(self.bezier) == 2:  # curves
                        if self.small_brush:
                            br = LSIZE_SMALL
                        else:
                            br = LSIZE
                        bezier(
                            self.img,
                            self.cols[self.col],
                            br,
                            self.bezier + [pygame.mouse.get_pos()],
                        )
                        self.bezier = []
                    if self.tool == T_FIL and not ra:  # flood fill
                        fill2(self.img, pygame.mouse.get_pos(), self.cols[self.col])

                elif event.button == 1 and not self.hide:
                    xp, yp = pygame.mouse.get_pos()
                    xp -= RES[0]
                    yp -= TOOL_HEIGHT
                    xp = xp // PALBW
                    yp = yp // PALBW
                    c = PALROWS * xp + yp
                    if yp >= 0 and c < len(self.cols):
                        self.col = c
                        self.getcolpic()

                    if yp == -3 and xp == 0:  # MAKE MOUNTAIN
                        # TODO: MAKE MOUNTAIN LOGIC
                        pass
                    if yp == -3 and xp == 1:  # undo
                        if len(self.undo) >= 2:
                            self.img = self.undo[-2].copy()
                            self.undo = [self.undo[-1], self.undo[-2]]
                    if yp == -3 and xp == 2:  # clear
                        self.undo.append(self.img.copy())
                        self.img.fill(0xFFFFFF)

                    if yp == -2 and xp == 0:
                        self.tool = 0
                        self.title()
                    if yp == -2 and xp == 1:
                        self.tool = 1
                        self.title()
                    if yp == -2 and xp == 2:
                        self.tool = 2
                        self.line_start = 0, 0
                        self.title()

                    if yp == -1 and xp == 0:
                        self.tool = 3
                        self.bezier = []
                        self.title()
                    if yp == -1 and xp == 1:
                        self.tool = 4
                        self.title()
                    if yp == -1 and xp == 2:
                        self.tool = 5
                        self.title()

                elif event.button == 2:
                    self.small_brush = not self.small_brush
                    self.title()
                elif event.button == 3:
                    self.tool += 1
                    if self.tool > len(tname) - 1:
                        self.tool = 0
                    self.title()
                    self.line_start = 0, 0
                    self.bezier = []

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mdown = False
                    if (
                        self.tool == T_STR and pygame.mouse.get_pos()[0] < RES[0]
                    ):  # straight lines
                        if self.small_brush:
                            br = LSIZE_SMALL
                        else:
                            br = LSIZE
                        pygame.draw.line(
                            self.img,
                            self.cols[self.col],
                            self.line_start,
                            pygame.mouse.get_pos(),
                            width=br,
                        )
                    if (
                        self.tool == T_CUR
                        and pygame.mouse.get_pos()[0] < RES[0]
                        and len(self.bezier) == 1
                    ):  # curves
                        self.bezier.append(pygame.mouse.get_pos())
                    if pygame.mouse.get_pos()[0] < self.img.get_size()[0]:
                        self.undo.append(self.img.copy())
                        if len(self.undo) > 2:
                            self.undo = self.undo[-2:]
            if event.type == pygame.MOUSEWHEEL:
                self.col -= event.y
                self.col = self.col % len(self.cols)
                self.getcolpic()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.palnum += 1
                if self.palnum >= len(palettes):
                    self.palnum = 0
                self.cols = palettes[self.palnum][0]
                self.col = 0
                self.getcolpic()
                self.title()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                self.tool += 1
                if self.tool > len(tname) - 1:
                    self.tool = 0
                self.title()
                self.line_start = 0, 0
                self.bezier = []
            if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                self.small_brush = not self.small_brush
                self.title()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                self.hide = not self.hide
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                if len(self.undo) >= 2:
                    self.img = self.undo[-2].copy()
                    self.undo = [self.undo[-1], self.undo[-2]]
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                pos = pygame.mouse.get_pos()
                try:
                    c = self.img.get_at(pos)
                except:
                    pass
                else:
                    ch = c[0] * (256**2) + c[1] * 256 + c[2]
                    in_pal = False
                    for n, col in enumerate(self.cols):
                        if ch == col:
                            self.col = n
                            in_pal = True
                            self.getcolpic()
                    if not in_pal:
                        self.cols.append(ch)
                        self.col = len(self.cols) - 1
                        self.getcolpic()

    def run(self):
        self.running = True
        while self.running:
            self.clock.tick(50)
            self.events()
            self.update()
        pygame.quit()

    def getcolpic(self):
        "Draw palette surface"
        self.colpic.fill(COLBG)
        for x in range(PALHE):
            for y in range(PALROWS):
                i = PALROWS * x + y
                if i < len(self.cols):
                    pygame.draw.rect(
                        self.colpic, self.cols[i], [PALBW * x, PALBW * y, PALBW, PALBW]
                    )
                if i == self.col:
                    pygame.draw.rect(
                        self.colpic, 0, [PALBW * x, PALBW * y, PALBW, PALBW], width=2
                    )
                    pygame.draw.rect(
                        self.colpic,
                        "0xffffff",
                        [PALBW * x + 2, PALBW * y + 2, PALBW - 4, PALBW - 4],
                        width=4,
                    )
                    pygame.draw.rect(
                        self.colpic,
                        0,
                        [PALBW * x + 6, PALBW * y + 6, PALBW - 12, PALBW - 12],
                        width=2,
                    )

    def title(self):
        if self.small_brush:
            bb = ", small brush"
        else:
            bb = ", large brush"
        if self.tool == T_FIL:  # flood fill
            bb = ""
        pygame.display.set_caption(
            f"MountainMaker ({tname[self.tool]}, {palettes[self.palnum][1]} palette)"
        )

    def update(self):
        if self.mdown:
            if self.small_brush:
                brsize = BRUSH_SMALL
            else:
                brsize = BRUSH
            x, y = pygame.mouse.get_pos()
            if self.tool == T_DOT:  # pen (aka dotted freehand in Deluxe Paint)
                pygame.draw.rect(
                    self.img,
                    self.cols[self.col],
                    [x - brsize // 2, y - brsize // 2, brsize, brsize],
                )
            elif self.tool == T_CON:  # pen (lines)
                steps = max(1, abs(x - self.lastpos[0]), abs(y - self.lastpos[1]))
                for n in range(steps):
                    xp = self.lastpos[0] + n * (x - self.lastpos[0]) / steps
                    yp = self.lastpos[1] + n * (y - self.lastpos[1]) / steps
                    pygame.draw.rect(
                        self.img,
                        self.cols[self.col],
                        [xp - brsize // 2, yp - brsize // 2, brsize, brsize],
                    )
                self.lastpos = x, y
            elif self.tool == T_AIR:  # airbrush
                for n in range(AIRDENS):
                    phi = random.uniform(0, 2 * math.pi)
                    r = random.gauss(0, brsize)
                    if abs(r) <= 2 * brsize:
                        pygame.draw.rect(
                            self.img,
                            self.cols[self.col],
                            [
                                x + r * math.sin(phi),
                                y + r * math.cos(phi),
                                AIRSIZE,
                                AIRSIZE,
                            ],
                        )

        self.screen.fill(COLBG)
        self.screen.blit(self.img, (0, 0))

        # draw tool previews
        if self.tool == T_STR and self.mdown:  # straight lines
            if self.small_brush:
                br = LSIZE_SMALL
            else:
                br = LSIZE
            pygame.draw.line(
                self.screen,
                self.cols[self.col],
                self.line_start,
                pygame.mouse.get_pos(),
                width=br,
            )

        if self.tool == T_CUR and len(self.bezier) == 1:  # curves
            if self.small_brush:
                br = LSIZE_SMALL
            else:
                br = LSIZE
            pygame.draw.line(
                self.screen,
                self.cols[self.col],
                self.bezier[0],
                pygame.mouse.get_pos(),
                width=br,
            )

        if self.tool == T_CUR and len(self.bezier) == 2:  # curves
            if self.small_brush:
                br = LSIZE_SMALL
            else:
                br = LSIZE
            bezier(
                self.screen,
                self.cols[self.col],
                br,
                self.bezier + [pygame.mouse.get_pos()],
            )

        # do flood fill steps if necessary
        if ra:
            for q in range(FILL_SPEED):
                do_fill(self.img)

        # draw toolbox
        if not self.hide:
            # if self.small_brush:
            #     self.screen.blit(brico_s, (RES[0], 0))
            # else:
            #     self.screen.blit(brico_b, (RES[0], 0))
            self.screen.blit(b_mtn, (RES[0], 0))
            self.screen.blit(b_undo, (RES[0] + 50, 0))
            self.screen.blit(b_clear, (RES[0] + 100, 0))

            self.screen.blit(is_act(0, self.tool), (RES[0], 50))
            self.screen.blit(is_act(1, self.tool), (RES[0] + 50, 50))
            self.screen.blit(is_act(2, self.tool), (RES[0] + 100, 50))

            # self.screen.blit(is_act(3, self.tool), (RES[0], 100))
            # self.screen.blit(is_act(4, self.tool), (RES[0] + 50, 100))
            # self.screen.blit(is_act(5, self.tool), (RES[0] + 100, 100))

            self.screen.blit(self.colpic, (RES[0], 100))
        pygame.display.flip()


c = Paint()
c.run()
