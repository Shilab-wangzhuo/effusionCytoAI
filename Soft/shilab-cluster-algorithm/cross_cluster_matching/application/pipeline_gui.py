import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import subprocess
import threading
import sys
import time
from datetime import datetime

CONFIG_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_gui_config.json")
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py")

BG        = "#1e1e2e"
PANEL     = "#2a2a3e"
ACCENT    = "#7c6af7"
ACCENT2   = "#56cfb2"
TEXT      = "#cdd6f4"
TEXT_DIM  = "#6c7086"
ENTRY_BG  = "#313244"
BTN_RUN   = "#40a02b"
BTN_STOP  = "#e64553"
BTN_SAVE  = "#7c6af7"
BTN_RST   = "#fe640b"
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_HEAD  = ("Segoe UI", 10, "bold")
FONT_BODY  = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)

SUPPORTED_MODELS = [
    "AlexNet", "ResNeXt", "ConvNextLarge", "GoogLeNet",
    "ResNet50", "ResNet101", "DenseNet121", "DenseNet161",
    "VGG16", "InceptionV3", "EfficientNetB0", "EfficientNetV2Large",
    "MobileNetV2", "MobileNetV3", "RegNet", "ShuffleNetV2",
    "SqueezeNet", "ViT_B16", "ViT_L16"
]

DEFAULT_RULES = (
    "0 | 4 9 | 0.5 | 3 | none\n"
    "4 | 0 9 | 0.5 | 3 | none\n"
    "9 | 0 4 | 0.5 | 3 | none"
)

def styled_entry(parent, textvariable=None, width=40):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                    relief="flat", font=FONT_BODY, bd=4)

def styled_btn(parent, text, command, color=ACCENT, fg="white", width=14):
    return tk.Button(parent, text=text, command=command,
                     bg=color, fg=fg, activebackground=color,
                     activeforeground="white", relief="flat",
                     font=FONT_HEAD, width=width, cursor="hand2", bd=0,
                     padx=6, pady=4)

def section_label(parent, text):
    f = tk.Frame(parent, bg=BG)
    tk.Label(f, text=text, bg=BG, fg=ACCENT, font=FONT_TITLE).pack(side="left")
    tk.Frame(f, bg=ACCENT, height=2).pack(side="left", fill="x", expand=True, padx=(8,0), pady=6)
    return f

def row_label(parent, text, width=22):
    return tk.Label(parent, text=text, bg=PANEL, fg=TEXT_DIM,
                    font=FONT_BODY, width=width, anchor="w")


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell Pipeline 控制台")
        self.geometry("1020x860")
        self.minsize(900, 700)
        self.configure(bg=BG)
        self.resizable(True, True)
        self._process = None
        self._vars = {}
        self._start_time = None
        self._timer_after_id = None
        self._build_ui()
        self._load_config()

    def _build_ui(self):
        header = tk.Frame(self, bg=ACCENT, height=48)
        header.pack(fill="x")
        tk.Label(header, text="  Cell Pipeline 控制台",
                 bg=ACCENT, fg="white", font=("Segoe UI", 14, "bold")).pack(side="left", padx=12, pady=8)
        tk.Label(header, text="conda env: learn",
                 bg=ACCENT, fg="#ddd", font=FONT_BODY).pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        left_wrap = tk.Frame(body, bg=BG)
        left_wrap.pack(side="left", fill="both", expand=True)

        canvas = tk.Canvas(left_wrap, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(left_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._scroll_frame = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)
        self._scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        sf = self._scroll_frame

        # Section 1: 路径配置
        section_label(sf, "  路径配置").pack(fill="x", padx=8, pady=(10,4))
        path_panel = tk.Frame(sf, bg=PANEL, bd=0)
        path_panel.pack(fill="x", padx=8, pady=2)
        for key, label, ftype in [
            ("positive_folder_path", "阳性病例文件夹",       "dir"),
            ("negative_folder_path", "阴性病例文件夹",       "dir"),
            ("malignant_cells_dir",  "参考域恶性细胞目录",   "dir"),
            ("benign_cells_dir",     "参考域良性细胞目录",   "dir"),
            ("model_path",           "预训练模型文件(.pth)", "file"),
            ("base_save_dir",        "结果保存目录",         "dir"),
        ]:
            self._path_row(path_panel, key, label, ftype)

        # Section 2: 模型与特征参数
        section_label(sf, "  模型与特征参数").pack(fill="x", padx=8, pady=(14,4))
        param_panel = tk.Frame(sf, bg=PANEL, bd=0)
        param_panel.pack(fill="x", padx=8, pady=2)
        defaults = {
            "model_type":                 "ResNeXt",
            "feature_layer":              "penultimate",
            "reference_k":                "10",
            "max_candidate_k":            "20",
            "pca_dim":                    "32",
            "batch_size":                 "16",
            "num_workers":                "0",
            "cell_size_weight":           "1.0",
            "mean":                       "",
            "std":                        "",
            "use_cell_level_rule_filter": "0",   # ← 新增，"1"=True / "0"=False
        }
        for i, (key, label, wtype, opts) in enumerate([
            ("model_type",                "模型类型",             "combo", SUPPORTED_MODELS),
            ("feature_layer",             "特征提取层",            "combo", ["penultimate","last","layer3","layer4"]),
            ("reference_k",               "参考域聚类数 K",        "entry", None),
            ("max_candidate_k",           "候选域最大 K",          "entry", None),
            ("pca_dim",                   "PCA 降维维度",          "entry", None),
            ("batch_size",                "Batch Size",            "entry", None),
            ("num_workers",               "Num Workers",           "entry", None),
            ("cell_size_weight",          "细胞大小权重",          "entry", None),
            ("mean",                      "归一化均值 (可选)",     "entry", None),
            ("std",                       "归一化标准差 (可选)",   "entry", None),
            ("use_cell_level_rule_filter","启用二层细胞过滤",      "check", None),  # ← 新增
        ]):
            self._param_row(param_panel, key, label, wtype, opts, defaults.get(key, ""), i)

        # Section 3: 恶性簇配置
        section_label(sf, "  恶性参考簇配置").pack(fill="x", padx=8, pady=(14,4))
        mal_panel = tk.Frame(sf, bg=PANEL, bd=0)
        mal_panel.pack(fill="x", padx=8, pady=2)
        row = tk.Frame(mal_panel, bg=PANEL)
        row.pack(fill="x", padx=10, pady=8)
        row_label(row, "恶性簇索引 (逗号分隔)").pack(side="left")
        self._vars["malignant_ref_clusters"] = tk.StringVar(value="0,4,9")
        styled_entry(row, textvariable=self._vars["malignant_ref_clusters"], width=30).pack(side="left", padx=6)
        tk.Label(row, text="  例如: 0,4,9  或  0,2", bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(side="left")

        # Section 4: 匹配规则
        section_label(sf, "  匹配规则配置").pack(fill="x", padx=8, pady=(14,4))
        rule_panel = tk.Frame(sf, bg=PANEL, bd=0)
        rule_panel.pack(fill="x", padx=8, pady=2)
        tk.Label(rule_panel,
                 text="每行一条规则，格式：target | avoid(空格分隔) | threshold_target | threshold_ratio | threshold_avoid_others\n"
                      "例：0 | 4 9 | 0.5 | 3 | none      （none 表示不设置该参数）",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_BODY, justify="left", wraplength=560
                 ).pack(anchor="w", padx=10, pady=(8,4))
        self._rule_text = tk.Text(rule_panel, height=5, bg=ENTRY_BG, fg=TEXT,
                                  insertbackground=TEXT, font=FONT_MONO, relief="flat", bd=4)
        self._rule_text.pack(fill="x", padx=10, pady=(0,10))
        self._rule_text.insert("1.0", DEFAULT_RULES)

        # 底部按钮栏
        btn_bar = tk.Frame(sf, bg=BG)
        btn_bar.pack(fill="x", padx=8, pady=12)
        styled_btn(btn_bar, "保存配置",      self._save_config,   color=BTN_SAVE).pack(side="left", padx=4)
        styled_btn(btn_bar, "导入配置",      self._import_config, color=ACCENT2, fg=BG).pack(side="left", padx=4)
        styled_btn(btn_bar, "重置参数",      self._reset,         color=BTN_RST).pack(side="left", padx=4)
        styled_btn(btn_bar, "运行 Pipeline", self._run,           color=BTN_RUN, width=18).pack(side="right", padx=4)
        self._stop_btn = styled_btn(btn_bar, "停止", self._stop,  color=BTN_STOP)
        self._stop_btn.pack(side="right", padx=4)
        self._stop_btn.config(state="disabled")

        # 右侧日志区
        right = tk.Frame(body, bg=BG, width=340)
        right.pack(side="right", fill="both", padx=(10,0))
        right.pack_propagate(False)

        # 日志头部（含清空 + 保存日志按钮）
        log_header = tk.Frame(right, bg=PANEL)
        log_header.pack(fill="x")
        tk.Label(log_header, text="运行日志", bg=PANEL, fg=ACCENT,
                 font=FONT_HEAD, pady=6).pack(side="left", padx=10)
        styled_btn(log_header, "保存日志", self._save_log,
                   color=BTN_SAVE, fg="white", width=8).pack(side="right", padx=4, pady=4)
        styled_btn(log_header, "清空", self._clear_log,
                   color=TEXT_DIM, fg=BG, width=6).pack(side="right", padx=2, pady=4)

        self._log = scrolledtext.ScrolledText(right, bg="#11111b", fg="#a6e3a1",
                                               font=FONT_MONO, relief="flat",
                                               state="disabled", wrap="word")
        self._log.pack(fill="both", expand=True, pady=(2,0))

        # 状态栏（含计时器显示）
        self._status_var = tk.StringVar(value="就绪")
        self._timer_var  = tk.StringVar(value="")
        status_bar = tk.Frame(self, bg=PANEL, height=24)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self._status_var, bg=PANEL,
                 fg=TEXT_DIM, font=FONT_BODY).pack(side="left", padx=10)
        tk.Label(status_bar, textvariable=self._timer_var, bg=PANEL,
                 fg=ACCENT2, font=FONT_BODY).pack(side="right", padx=10)

    # ── 路径 / 参数行构建 ──────────────────────────────────────────────────
    def _path_row(self, parent, key, label, ftype):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=10, pady=4)
        row_label(row, label).pack(side="left")
        self._vars[key] = tk.StringVar()
        styled_entry(row, textvariable=self._vars[key], width=36).pack(side="left", padx=6)
        cmd = (lambda k=key: self._browse_dir(k)) if ftype == "dir" else (lambda k=key: self._browse_file(k))
        styled_btn(row, "浏览", cmd, color=ACCENT2, fg=BG, width=8).pack(side="left")

    def _param_row(self, parent, key, label, wtype, opts, default, idx):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=10, pady=3)
        row_label(row, label).pack(side="left")
        self._vars[key] = tk.StringVar(value=default)
        if wtype == "combo":
            ttk.Combobox(row, textvariable=self._vars[key],
                         values=opts, width=22, state="readonly",
                         font=FONT_BODY).pack(side="left", padx=6)
        elif wtype == "check":
            # ── 复选框：用 StringVar "1"/"0" 驱动，保持与其他参数存储方式一致 ──
            cb = tk.Checkbutton(
                row,
                variable=self._vars[key],
                onvalue="1", offvalue="0",
                bg=PANEL, fg=TEXT,
                activebackground=PANEL, activeforeground=TEXT,
                selectcolor=ENTRY_BG,
                font=FONT_BODY,
                cursor="hand2",
                bd=0,
            )
            cb.pack(side="left", padx=6)
            # 勾选状态说明文字
            tk.Label(row, text="勾选表示在聚类匹配后，再用规则对单细胞余弦相似度进行二次过滤",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(side="left", padx=4)
        else:
            styled_entry(row, textvariable=self._vars[key], width=20).pack(side="left", padx=6)

    def _browse_dir(self, key):
        d = filedialog.askdirectory(title="选择文件夹")
        if d: self._vars[key].set(d)

    def _browse_file(self, key):
        f = filedialog.askopenfilename(title="选择文件",
            filetypes=[("模型文件", "*.pth *.pt"), ("所有文件", "*.*")])
        if f: self._vars[key].set(f)

    # ── 日志写入 ───────────────────────────────────────────────────────────
    def _log_write(self, msg, color=None):
        self._log.config(state="normal")
        if color:
            self._log.tag_config(color, foreground=color)
            self._log.insert("end", msg, color)
        else:
            self._log.insert("end", msg)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _save_log(self):
        content = self._log.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("提示", "日志为空，无需保存。")
            return
        default_name = "pipeline_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        path = filedialog.asksaveasfilename(
            title="保存日志",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("保存成功", f"日志已保存到:\n{path}")

    # ── 计时器 ─────────────────────────────────────────────────────────────
    def _start_timer(self):
        self._start_time = time.time()
        self._tick_timer()

    def _tick_timer(self):
        if self._start_time is None:
            return
        elapsed = int(time.time() - self._start_time)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self._timer_var.set(f"⏱ 已运行 {h:02d}:{m:02d}:{s:02d}")
        self._timer_after_id = self.after(1000, self._tick_timer)

    def _stop_timer(self):
        if self._timer_after_id:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._start_time = None
        return elapsed

    # ── 参数解析 ───────────────────────────────────────────────────────────
    def _parse_rules(self, malignant_clusters):
        lines = self._rule_text.get("1.0", "end").strip().splitlines()
        rules = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                raise ValueError(f"第 {i+1} 行规则格式错误（需要至少4段用 | 分隔）:\n{line}")
            target     = int(parts[0])
            avoid      = [int(x) for x in parts[1].split() if x]
            thr_target = float(parts[2])
            thr_ratio  = None if parts[3].lower() == "none" else float(parts[3])
            thr_avoid  = None
            if len(parts) >= 5:
                thr_avoid = None if parts[4].lower() == "none" else float(parts[4])
            rules.append({
                "target": target, "avoid": avoid,
                "threshold_target": thr_target,
                "threshold_ratio":  thr_ratio,
                "threshold_avoid_others": thr_avoid,
            })
        if not rules:
            raise ValueError("未填写任何匹配规则")
        return rules

    def _collect_params(self):
        v = {k: var.get().strip() for k, var in self._vars.items()}
        for rp in ["positive_folder_path", "negative_folder_path",
                   "malignant_cells_dir", "benign_cells_dir",
                   "model_path", "base_save_dir"]:
            if not v[rp]:
                raise ValueError(f"请填写：{rp}")
        mal_clusters = [int(x.strip()) for x in v["malignant_ref_clusters"].split(",") if x.strip()]
        if not mal_clusters:
            raise ValueError("恶性簇索引不能为空")
        rules = self._parse_rules(mal_clusters)
        if len(rules) != len(mal_clusters):
            raise ValueError(f"规则数量({len(rules)})必须与恶性簇数量({len(mal_clusters)})一致")
        mean_val = [float(x) for x in v["mean"].split(",")] if v["mean"] else None
        std_val  = [float(x) for x in v["std"].split(",")]  if v["std"]  else None
        return {
            "positive_folder_path":        v["positive_folder_path"],
            "negative_folder_path":        v["negative_folder_path"],
            "malignant_cells_dir":         v["malignant_cells_dir"],
            "benign_cells_dir":            v["benign_cells_dir"],
            "model_path":                  v["model_path"],
            "base_save_dir":               v["base_save_dir"],
            "model_type":                  v["model_type"],
            "feature_layer":               v["feature_layer"],
            "reference_k":                 int(v["reference_k"]),
            "max_candidate_k":             int(v["max_candidate_k"]),
            "pca_dim":                     int(v["pca_dim"]),
            "batch_size":                  int(v["batch_size"]),
            "num_workers":                 int(v["num_workers"]),
            "cell_size_weight":            float(v["cell_size_weight"]),
            "mean":                        mean_val,
            "std":                         std_val,
            "malignant_ref_clusters":      mal_clusters,
            "matching_rules":              rules,
            # "1" → True，"0" 或空 → False
            "use_cell_level_rule_filter":  v["use_cell_level_rule_filter"] == "1",
        }

    # ── 日志打印参数摘要 ───────────────────────────────────────────────────
    def _log_params(self, params, start_dt):
        sep = "=" * 50
        self._log_write(sep + "\n", "#89b4fa")
        self._log_write(f"  Pipeline 启动时间: {start_dt}\n", "#cba6f7")
        self._log_write(sep + "\n", "#89b4fa")
        self._log_write("【参数一览】\n", "#f9e2af")

        path_keys = [
            ("positive_folder_path", "阳性病例文件夹"),
            ("negative_folder_path", "阴性病例文件夹"),
            ("malignant_cells_dir",  "参考域恶性细胞目录"),
            ("benign_cells_dir",     "参考域良性细胞目录"),
            ("model_path",           "预训练模型文件"),
            ("base_save_dir",        "结果保存目录"),
        ]
        self._log_write("\n─── 路径配置 ───\n", "#89dceb")
        for k, label in path_keys:
            self._log_write(f"  {label:<18}: {params[k]}\n")

        self._log_write("\n─── 模型与特征参数 ───\n", "#89dceb")
        model_keys = [
            ("model_type",                "模型类型"),
            ("feature_layer",             "特征提取层"),
            ("reference_k",               "参考域聚类数 K"),
            ("max_candidate_k",           "候选域最大 K"),
            ("pca_dim",                   "PCA 降维维度"),
            ("batch_size",                "Batch Size"),
            ("num_workers",               "Num Workers"),
            ("cell_size_weight",          "细胞大小权重"),
            ("mean",                      "归一化均值"),
            ("std",                       "归一化标准差"),
            ("use_cell_level_rule_filter","启用二层细胞过滤"),  # ← 新增
        ]
        for k, label in model_keys:
            val = params[k] if params[k] is not None else "（未设置）"
            self._log_write(f"  {label:<18}: {val}\n")

        self._log_write("\n─── 恶性参考簇 ───\n", "#89dceb")
        self._log_write(f"  恶性簇索引           : {params['malignant_ref_clusters']}\n")

        self._log_write("\n─── 匹配规则 ───\n", "#89dceb")
        for i, rule in enumerate(params["matching_rules"]):
            self._log_write(
                f"  规则 {i+1}: target={rule['target']}  avoid={rule['avoid']}  "
                f"thr_target={rule['threshold_target']}  "
                f"thr_ratio={rule['threshold_ratio']}  "
                f"thr_avoid_others={rule['threshold_avoid_others']}\n"
            )

        self._log_write("\n" + sep + "\n", "#89b4fa")
        self._log_write("  开始执行 Pipeline...\n", "#a6e3a1")
        self._log_write(sep + "\n", "#89b4fa")

    # ── 构建临时运行脚本 ───────────────────────────────────────────────────
    def _build_runner_script(self, params):
        lines = [
            "import sys, os",
            f"sys.path.insert(0, r'{os.path.dirname(PIPELINE_PATH)}')",
            "from pipeline import run_pipeline",
            "run_pipeline(",
            f"    positive_folder_path        = r'{params['positive_folder_path']}',",
            f"    negative_folder_path        = r'{params['negative_folder_path']}',",
            f"    malignant_cells_dir         = r'{params['malignant_cells_dir']}',",
            f"    benign_cells_dir            = r'{params['benign_cells_dir']}',",
            f"    model_path                  = r'{params['model_path']}',",
            f"    base_save_dir               = r'{params['base_save_dir']}',",
            f"    model_type                  = '{params['model_type']}',",
            f"    feature_layer               = '{params['feature_layer']}',",
            f"    reference_k                 = {params['reference_k']},",
            f"    max_candidate_k             = {params['max_candidate_k']},",
            f"    pca_dim                     = {params['pca_dim']},",
            f"    batch_size                  = {params['batch_size']},",
            f"    num_workers                 = {params['num_workers']},",
            f"    cell_size_weight            = {params['cell_size_weight']},",
            f"    mean                        = {params['mean']},",
            f"    std                         = {params['std']},",
            f"    malignant_ref_clusters      = {params['malignant_ref_clusters']},",
            f"    matching_rules              = {params['matching_rules']},",
            f"    use_cell_level_rule_filter  = {params['use_cell_level_rule_filter']},",  # ← 新增
            ")",
        ]
        return "\n".join(lines)

    # ── 运行 Pipeline ──────────────────────────────────────────────────────
    def _run(self):
        try:
            params = self._collect_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pipeline_runner_tmp.py")
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(self._build_runner_script(params))

        start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_params(params, start_dt)
        self._start_timer()
        self._status_var.set("运行中...")
        self._stop_btn.config(state="normal")

        def worker():
            try:
                cmd = ["conda", "run", "-n", "learn", "--no-capture-output",
                       "python", runner_path]
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                for line in self._process.stdout:
                    self.after(0, self._log_write, line)
                self._process.wait()
                rc = self._process.returncode

                elapsed = self._stop_timer()
                h, rem = divmod(int(elapsed), 3600)
                m, s   = divmod(rem, 60)
                elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
                end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                sep = "=" * 50
                if rc == 0:
                    def _done():
                        self._log_write("\n" + sep + "\n", "#89b4fa")
                        self._log_write(f"  Pipeline 运行完成！\n", "#a6e3a1")
                        self._log_write(f"  结束时间: {end_dt}\n", "#cba6f7")
                        self._log_write(f"  总耗时:   {elapsed_str}\n", "#f9e2af")
                        self._log_write(sep + "\n", "#89b4fa")
                        self._status_var.set(f"完成  ✓  耗时 {elapsed_str}")
                        self._timer_var.set(f"⏱ 总耗时 {elapsed_str}")
                    self.after(0, _done)
                else:
                    def _fail():
                        self._log_write("\n" + sep + "\n", "#89b4fa")
                        self._log_write(f"  Pipeline 异常退出，返回码: {rc}\n", "#f38ba8")
                        self._log_write(f"  结束时间: {end_dt}\n", "#cba6f7")
                        self._log_write(f"  已运行:   {elapsed_str}\n", "#f9e2af")
                        self._log_write(sep + "\n", "#89b4fa")
                        self._status_var.set(f"失败 (code {rc})  耗时 {elapsed_str}")
                        self._timer_var.set(f"⏱ 总耗时 {elapsed_str}")
                    self.after(0, _fail)

            except Exception as ex:
                elapsed = self._stop_timer()
                self.after(0, self._log_write, f"\n启动失败: {ex}\n", "#f38ba8")
                self.after(0, self._status_var.set, "启动失败")
                self.after(0, self._timer_var.set, "")
            finally:
                self.after(0, self._stop_btn.config, {"state": "disabled"})
                self._process = None

        threading.Thread(target=worker, daemon=True).start()

    # ── 停止（Windows 专用：taskkill 杀掉整棵进程树）─────────────────────
    def _stop(self):
        if self._process:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            elapsed = self._stop_timer()
            h, rem = divmod(int(elapsed), 3600)
            m, s   = divmod(rem, 60)
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
            self._log_write(f"\n已手动停止  已运行: {elapsed_str}\n", "#f38ba8")
            self._status_var.set(f"已停止  耗时 {elapsed_str}")
            self._timer_var.set(f"⏱ 总耗时 {elapsed_str}")
            self._stop_btn.config(state="disabled")

    # ── 配置保存 / 导入 / 加载 / 重置 ────────────────────────────────────
    def _save_config(self):
        default_name = "pipeline_gui_config_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        path = filedialog.asksaveasfilename(
            title="保存配置",
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON 配置文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        cfg = {k: var.get() for k, var in self._vars.items()}
        cfg["__rules__"] = self._rule_text.get("1.0", "end").strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("保存成功", f"配置已保存到:\n{path}")

    def _import_config(self):
        path = filedialog.askopenfilename(
            title="选择配置文件",
            defaultextension=".json",
            filetypes=[("JSON 配置文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if k == "__rules__":
                    self._rule_text.delete("1.0", "end")
                    self._rule_text.insert("1.0", v)
                elif k in self._vars:
                    self._vars[k].set(v)
            messagebox.showinfo("导入成功", f"配置已从以下文件导入:\n{path}")
        except Exception as e:
            messagebox.showerror("导入失败", f"读取配置文件时出错:\n{e}")

    def _load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if k == "__rules__":
                    self._rule_text.delete("1.0", "end")
                    self._rule_text.insert("1.0", v)
                elif k in self._vars:
                    self._vars[k].set(v)
        except Exception:
            pass

    def _reset(self):
        if not messagebox.askyesno("确认", "确定要重置所有参数吗？"):
            return
        for var in self._vars.values():
            var.set("")
        self._rule_text.delete("1.0", "end")
        self._rule_text.insert("1.0", DEFAULT_RULES)
        self._vars["model_type"].set("ResNeXt")
        self._vars["feature_layer"].set("penultimate")
        self._vars["reference_k"].set("10")
        self._vars["max_candidate_k"].set("20")
        self._vars["pca_dim"].set("32")
        self._vars["batch_size"].set("16")
        self._vars["num_workers"].set("0")
        self._vars["cell_size_weight"].set("1.0")
        self._vars["malignant_ref_clusters"].set("0,4,9")
        self._vars["use_cell_level_rule_filter"].set("0")   # ← 新增，默认不使用细胞级规则过滤


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()