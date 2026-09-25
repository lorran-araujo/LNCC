"""Desenha o diagrama de classes UML da PIMUCS com layout manual (matplotlib).

Uso:  python3 desenhar_classes.py   ->  diagrama_classes.png / .svg / .pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

W, H, DPI = 40, 26, 110
fig = plt.figure(figsize=(W, H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
R = fig.canvas.get_renderer()

FS, LH, PADX, PADY = 10.5, 0.235, 0.16, 0.10   # fonte, altura de linha, paddings
BORDER = "#222222"


def tw(s, fs=FS, weight="normal", style="normal"):
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(R).width / DPI
    t.remove()
    return w


class Box:
    def __init__(self, name, attrs=(), ops=(), x=0, y=0, stereo=None, abstract=False, fill="white"):
        self.name, self.attrs, self.ops = name, list(attrs), list(ops)
        self.stereo, self.abstract, self.fill = stereo, abstract, fill
        self.x, self.y = x, y
        lines = self.attrs + self.ops
        self.w = max([tw(l) for l in lines] + [tw(name, weight="bold") , tw(stereo or "")]) + 2 * PADX + 0.1
        self.h_head = LH * (1 + (1 if stereo else 0)) + 2 * PADY
        self.h_attr = LH * max(len(self.attrs), 1) + 2 * PADY * (1 if self.attrs else 0.6)
        self.h_ops = LH * max(len(self.ops), 1) + 2 * PADY * (1 if self.ops else 0.6)
        self.h = self.h_head + self.h_attr + self.h_ops

    # âncoras (fração ao longo do lado)
    def top(self, f=0.5): return (self.x + f * self.w, self.y)
    def bottom(self, f=0.5): return (self.x + f * self.w, self.y + self.h)
    def left(self, f=0.5): return (self.x, self.y + f * self.h)
    def right(self, f=0.5): return (self.x + self.w, self.y + f * self.h)
    @property
    def cx(self): return self.x + self.w / 2

    def draw(self):
        ax.add_patch(Rectangle((self.x, self.y), self.w, self.h, fc=self.fill, ec=BORDER, lw=1.4, zorder=3))
        y = self.y + PADY
        if self.stereo:
            ax.text(self.cx, y + LH / 2, f"«{self.stereo}»", ha="center", va="center", fontsize=FS - 1.5, zorder=4)
            y += LH
        ax.text(self.cx, y + LH / 2, self.name, ha="center", va="center", fontsize=FS + 0.5,
                fontweight="bold", fontstyle="italic" if self.abstract else "normal", zorder=4)
        y1 = self.y + self.h_head
        y2 = y1 + self.h_attr
        for yy in (y1, y2):
            ax.plot([self.x, self.x + self.w], [yy, yy], color=BORDER, lw=1.2, zorder=4)
        for i, l in enumerate(self.attrs):
            ax.text(self.x + PADX, y1 + PADY + LH * (i + 0.5), l, va="center", fontsize=FS, zorder=4)
        for i, l in enumerate(self.ops):
            ax.text(self.x + PADX, y2 + PADY + LH * (i + 0.5), l, va="center", fontsize=FS, zorder=4)


def package(title, boxes, color, pad=0.45, top=0.85, extra=None):
    x0 = min(b.x for b in boxes) - pad
    y0 = min(b.y for b in boxes) - top
    x1 = max(b.x + b.w for b in boxes) + pad
    y1 = max(b.y + b.h for b in boxes) + pad
    if extra:
        x0, y0, x1, y1 = min(x0, extra[0]), min(y0, extra[1]), max(x1, extra[2]), max(y1, extra[3])
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=color, ec="#555555", lw=1.6, zorder=1))
    ax.text(x0 + 0.3, y0 + 0.42, title, fontsize=FS + 4, fontweight="bold", va="center", zorder=2)
    return (x0, y0, x1, y1)


def arrow(pts, kind="assoc", dashed=False, label=None, lpos=None, ha="left", color="#111111"):
    """pts: lista de pontos (x,y); a ponta fica no último."""
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    dx, dy = xb - xa, yb - ya
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    size = 0.30
    if kind == "gen":
        base = (xb - ux * size * 1.5, yb - uy * size * 1.5)
        poly = Polygon([(xb, yb), (base[0] + px * size * 0.8, base[1] + py * size * 0.8),
                        (base[0] - px * size * 0.8, base[1] - py * size * 0.8)],
                       closed=True, fc="white", ec=color, lw=1.4, zorder=6)
        ax.add_patch(poly)
        end = base
    else:
        end = (xb, yb)
        for s in (1, -1):
            ax.plot([xb, xb - ux * size + px * s * 0.14], [yb, yb - uy * size + py * s * 0.14],
                    color=color, lw=1.4, zorder=6)
    xs = [p[0] for p in pts[:-1]] + [end[0]]
    ys = [p[1] for p in pts[:-1]] + [end[1]]
    ax.plot(xs, ys, color=color, lw=1.4, ls=(0, (5, 3)) if dashed else "-", zorder=5, solid_capstyle="butt")
    if label:
        lx, ly = lpos if lpos else (xb + 0.10, yb - 0.30)
        ax.text(lx, ly, label, fontsize=FS - 1.5, ha=ha, va="center", zorder=7, color="#111111",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


# ---------------------------------------------------------------- classes
fillE = "#fff2e0"; fillM = "#fffde7"
PS = Box("PipelineStage", ops=["+ execute(): bool"], stereo="abstract", abstract=True)
IS = Box("IngestionScheduler", ops=["+ check_for_new_data(): bool"])
PM = Box("PipelineManager", attrs=["+ failed_stage: str | None"], ops=["+ execute_pipeline(): bool"])
DP = Box("DataPreProcessing", ops=["+ execute(): bool",
                                   "+ preprocess(images: list[GeoReferencedImage]): dict"])
TR = Box("Training", ops=["+ execute(): bool", "+ train(model: AIModel, data: dict | None): AIModel"])
GE = Box("Generate", ops=["+ execute(): bool", "+ generate_map(model: AIModel, data: dict | None):",
                          "      LandUseCoverMap"])
MO = Box("Monitoring", attrs=["+ simulate_irregularity: bool"],
         ops=["+ execute(): bool", "+ compare_maps(land_use_map: LandUseCoverMap,",
              "      zoning_map: ZoningMap): list[IrregularityAlert]"])
DR = Box("DataPreprocessedRepository", attrs=["+ data: dict | None"],
         ops=["+ save(data: dict): None", "+ load(): dict | None"])
MR = Box("ModelRepository", ops=["+ load(): AIModel", "+ save(model: AIModel): None"])
EX = Box("ExternalDataRepository", attrs=["+ has_new_data: bool"],
         ops=["+ check_new_data(): bool", "+ get_new_images(): list[GeoReferencedImage]"], fill=fillE)
LM = Box("LandUseCoverManagement", ops=["+ update_map(land_use_map: LandUseCoverMap): None",
                                        "+ get_current_map(): LandUseCoverMap | None"])
ZM = Box("ZoningManagement", ops=["+ get_zoning_map(): ZoningMap"])
AM = Box("AlertManagement", ops=["+ register_alert(alert: IrregularityAlert): None",
                                 "+ get_alerts(): list[IrregularityAlert]"])
CA = Box("ControlledAccess", ops=["+ view_alerts(): list[IrregularityAlert]"])
GI = Box("GeoReferencedImage", attrs=["+ name: str", "+ acquisition_date: datetime"], fill=fillM)
AI = Box("AIModel", attrs=["+ version: int"], fill=fillM)
LC = Box("LandUseCoverMap", attrs=["+ version: int", "+ model_version: int"], fill=fillM)
ZP = Box("ZoningMap", attrs=["+ version: int"], fill=fillM)
IA = Box("IrregularityAlert", attrs=["+ description: str"], fill=fillM)

# ---------------------------------------------------------------- layout
GAP = 0.7
XIA = 7.2
Y_R1, Y_R2 = 2.5, 5.9
DP.x, DP.y = XIA, Y_R2
TR.x, TR.y = DP.x + DP.w + GAP, Y_R2
GE.x, GE.y = TR.x + TR.w + GAP, Y_R2
MO.x, MO.y = GE.x + GE.w + GAP, Y_R2
IS.x, PM.y = XIA, Y_R1
PM.x = IS.x + IS.w + 1.4
IS.y = PM.y + PM.h / 2 - IS.h / 2
PS.x = GE.cx - PS.w / 2
PS.y = PM.y + PM.h / 2 - PS.h / 2

bot = Y_R2 + max(DP.h, TR.h, GE.h, MO.h)
Y_R3 = bot + 2.75
DR.x, DR.y = XIA + 0.5, Y_R3
MR.x, MR.y = DR.x + DR.w + 1.5, Y_R3

ia_boxes = [IS, PM, PS, DP, TR, GE, MO, DR, MR]
ia_bottom = Y_R3 + max(DR.h, MR.h) + 0.45
ia_rect_right = MO.x + MO.w + 0.45

Y_DM = ia_bottom + 3.0
CA.x, CA.y = ia_rect_right + 1.5, Y_R2
AM.x, AM.y = CA.cx - AM.w / 2, Y_DM
ZM.x, ZM.y = AM.x - ZM.w - 0.9, Y_DM
LM.x, LM.y = ZM.x - LM.w - 0.9, Y_DM
Y_MD = Y_DM + max(LM.h, ZM.h, AM.h) + 2.5

EX.x, EX.y = 0.9, 4.3
GI.x, GI.y = EX.cx - GI.w / 2, Y_MD
AI.x, AI.y = MR.cx - AI.w / 2, Y_MD
LC.x, LC.y = LM.cx - LC.w / 2, Y_MD
ZP.x, ZP.y = ZM.cx - ZP.w / 2, Y_MD
IA.x, IA.y = AM.cx - IA.w / 2, Y_MD

rects = [
    package("IAProcessing", ia_boxes, "#dbe8fb"),
    package("Sistema externo", [EX], "#ffe6cc", pad=0.4),
    package("DataManagement", [LM, ZM, AM], "#d5e8d4"),
    package("WebApplication", [CA], "#e6d0e6"),
    package("Models (entidades de domínio)", [GI, AI, LC, ZP, IA], "#fffbd6"),
]
for b in (PS, IS, PM, DP, TR, GE, MO, DR, MR, EX, LM, ZM, AM, CA, GI, AI, LC, ZP, IA):
    b.draw()

# ---------------------------------------------------------------- relações
# generalização: barramento -> PipelineStage
ybus = Y_R2 - 1.15
for st in (DP, TR, MO):
    ax.plot([st.cx, st.cx], [Y_R2, ybus], color="#111", lw=1.4, zorder=5)
ax.plot([DP.cx, MO.cx], [ybus, ybus], color="#111", lw=1.4, zorder=5)
arrow([(GE.cx, Y_R2), (PS.cx, PS.y + PS.h)], kind="gen")

arrow([PM.right(0.5), PS.left(0.5)], label="- stages   1..*", lpos=(PM.x + PM.w + 0.2, PM.y + PM.h / 2 - 0.3))
arrow([IS.right(0.5), PM.left(0.5)])

# externo
lane1, lane2 = XIA - 1.05, XIA - 0.55
e_r = EX.x + EX.w
ex_y1, ex_y2 = EX.y + EX.h * 0.30, EX.y + EX.h * 0.72
arrow([IS.left(0.5), (lane1, IS.y + IS.h / 2), (lane1, ex_y1), (e_r, ex_y1)])
dpy = DP.y + DP.h_head / 2 + 0.35
arrow([(DP.x, dpy), (lane2, dpy), (lane2, ex_y2), (e_r, ex_y2)])
arrow([EX.bottom(0.5), GI.top(0.5)], dashed=True, label="«create»",
      lpos=(EX.cx + 0.12, (EX.y + EX.h + GI.y) / 2), color="#555")

# etapas -> repositórios do IAProcessing
lanes = [bot + 0.40 + 0.42 * i for i in range(5)]
def elbow(src, dst, lane):
    arrow([src, (src[0], lane), (dst[0], lane), dst])
d1, d2, d3 = DR.top(0.20), DR.top(0.55), DR.top(0.88)
m1, m2 = MR.top(0.25), MR.top(0.75)
elbow(DP.bottom(0.5), d1, lanes[0])
elbow(TR.bottom(0.30), d2, lanes[1])
elbow(TR.bottom(0.70), m1, lanes[2])
elbow(GE.bottom(0.12), d3, lanes[3])
elbow(GE.bottom(0.32), m2, lanes[4])

# etapas -> DataManagement
elbow_ys = [Y_DM - 0.75, Y_DM - 1.2, Y_DM - 1.65, Y_DM - 2.1]
arrow([GE.bottom(0.88), (GE.x + 0.88 * GE.w, elbow_ys[0]), (LM.x + 0.75 * LM.w, elbow_ys[0]), LM.top(0.75)])
mx = [MO.x + f * MO.w for f in (0.15, 0.5, 0.85)]
arrow([(mx[0], MO.y + MO.h), (mx[0], elbow_ys[1]), (LM.x + 0.30 * LM.w, elbow_ys[1]), LM.top(0.30)])
arrow([(mx[1], MO.y + MO.h), (mx[1], elbow_ys[2]), (ZM.cx, elbow_ys[2]), ZM.top(0.5)])
arrow([(mx[2], MO.y + MO.h), (mx[2], elbow_ys[3]), (AM.x + 0.30 * AM.w, elbow_ys[3]), AM.top(0.30)])
arrow([CA.bottom(0.5), (CA.cx, AM.y)])

# gerenciadores/repositórios -> entidades
mid = lambda a, b: (a.y + a.h + b.y) / 2
arrow([MR.bottom(0.5), AI.top(0.5)], label="- model", lpos=(MR.cx + 0.12, mid(MR, AI)))
arrow([LM.bottom(0.5), LC.top(0.5)], label="- current_map   0..1", lpos=(LM.cx + 0.12, mid(LM, LC)))
arrow([ZM.bottom(0.5), ZP.top(0.5)], label="- zoning_map", lpos=(ZM.cx + 0.12, mid(ZM, ZP)))
arrow([AM.bottom(0.5), IA.top(0.5)], label="- alerts   0..*", lpos=(AM.cx + 0.12, mid(AM, IA)))

# ---------------------------------------------------------------- legenda
lx, ly = 3.9, Y_DM - 0.4
lw_, lh_ = 8.6, 3.5
ax.add_patch(FancyBboxPatch((lx, ly), lw_, lh_, boxstyle="round,pad=0,rounding_size=0.15",
                            fc="white", ec="#999999", lw=1.2, zorder=1))
ax.text(lx + 0.25, ly + 0.35, "Legenda", fontsize=FS + 1.5, fontweight="bold", va="center")
arrow([(lx + 0.3, ly + 0.95), (lx + 1.4, ly + 0.95)], kind="gen")
ax.text(lx + 1.7, ly + 0.95, "generalização (herança)", fontsize=FS - 0.5, va="center")
arrow([(lx + 0.3, ly + 1.55), (lx + 1.4, ly + 1.55)])
ax.text(lx + 1.7, ly + 1.55, "associação unidirecional (sem multiplicidade = 1)", fontsize=FS - 0.5, va="center")
arrow([(lx + 0.3, ly + 2.15), (lx + 1.4, ly + 2.15)], dashed=True, color="#555")
ax.text(lx + 1.7, ly + 2.15, "dependência «create»", fontsize=FS - 0.5, va="center")
ax.text(lx + 0.3, ly + 2.85,
        "Nomes de atributo omitidos nas associações com multiplicidade 1: derivam do tipo\n"
        "de destino (ex.: model_repository → ModelRepository). Usos de Models em assinaturas: ver pacotes.",
        fontsize=FS - 1.8, va="center", linespacing=1.3, color="#333333")

ax.text(0.6, 0.55, "PIMUCS - Diagrama de classes (IAProcessing e módulos de apoio)", fontsize=FS + 7,
        fontweight="bold", va="center")

xmax = max(r[2] for r in rects) + 0.5
ymax = max(r[3] for r in rects) + 0.5
ax.set_xlim(0, xmax); ax.set_ylim(ymax, 0)
fig.set_size_inches(xmax, ymax)
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"diagrama_classes.{ext}", dpi=DPI if ext == "png" else None)
print("ok", round(xmax, 1), round(ymax, 1))
