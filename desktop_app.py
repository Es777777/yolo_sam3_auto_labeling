"""Native Tk desktop application for text prompted SAM3 labeling."""

from contextlib import nullcontext
import copy
import gc
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile

from PIL import Image, ImageOps, ImageTk
import yaml

from label_tool import ALIASES, EXTENSIONS, parse_prompts

ROOT = Path(__file__).resolve().parent
COLORS = ["#12a77d", "#ee9b24", "#4389e8", "#df5b75", "#956bd0", "#20abb4"]


def archive_members(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for info in archive.infolist():
            name = Path(info.filename)
            if info.is_dir() or name.is_absolute() or ".." in name.parts:
                continue
            if name.suffix.lower() in EXTENSIONS:
                members.append(info)
        return members


def discover_images(source, is_zip=False):
    if is_zip:
        return [Path(info.filename) for info in archive_members(source)]
    base = Path(source).resolve()
    return sorted(p.relative_to(base) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


class DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM3 文本自动标注工作台")
        self.root.geometry("1380x860")
        self.root.minsize(1050, 700)
        self.queue = queue.Queue()
        self.stop_requested = False
        self.source = None
        self.source_is_zip = False
        self.output = None
        self.classes = []
        self.items = []
        self.current = -1
        self.image = None
        self.tk_image = None
        self.overlay = []
        self.dirty = False
        self._build_style()
        self._build_ui()
        self.root.after(120, self._drain_queue)

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"), foreground="#17362b")
        style.configure("Section.TLabel", font=("TkDefaultFont", 11, "bold"), foreground="#147d63")
        style.configure("Accent.TButton", foreground="white", background="#147d63", padding=(12, 8))
        style.map("Accent.TButton", background=[("active", "#0f684f")])

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(18, 14)); header.pack(fill="x")
        ttk.Label(header, text="SAM3 文本自动标注工作台", style="Title.TLabel").pack(side="left")
        self.status = ttk.Label(header, text="请选择图集", foreground="#718079"); self.status.pack(side="right")
        body = ttk.Panedwindow(self.root, orient="horizontal"); body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left = ttk.Frame(body, padding=12); center = ttk.Frame(body, padding=12); right = ttk.Frame(body, padding=12)
        body.add(left, weight=0); body.add(center, weight=1); body.add(right, weight=0)
        self._build_left(left); self._build_center(center); self._build_right(right)

    def _build_left(self, frame):
        ttk.Label(frame, text="1  图集来源", style="Section.TLabel").pack(anchor="w", pady=(0, 9))
        source_row = ttk.Frame(frame); source_row.pack(fill="x")
        ttk.Button(source_row, text="选择文件夹", command=self.choose_folder).pack(side="left", fill="x", expand=True)
        ttk.Button(source_row, text="选择 ZIP", command=self.choose_zip).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.source_var = tk.StringVar(); ttk.Entry(frame, textvariable=self.source_var, state="readonly").pack(fill="x", pady=8)
        ttk.Button(frame, text="输出目录", command=self.choose_output).pack(anchor="w")
        self.output_var = tk.StringVar(); ttk.Entry(frame, textvariable=self.output_var, state="readonly").pack(fill="x", pady=8)
        ttk.Separator(frame).pack(fill="x", pady=14)
        ttk.Label(frame, text="2  目标文本", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.prompt_text = tk.Text(frame, height=8, width=28, wrap="word", undo=True)
        self.prompt_text.insert("1.0", "person = person\n汽车 = car")
        self.prompt_text.pack(fill="both", expand=False)
        ttk.Label(frame, text="每行一个类别：类别名 = 英文描述\n同类提示词用 | 分隔。", foreground="#77837b").pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(frame); row.pack(fill="x", pady=(14, 2))
        ttk.Label(row, text="置信度").pack(side="left"); self.threshold = tk.DoubleVar(value=.35)
        ttk.Label(row, textvariable=self.threshold).pack(side="right")
        ttk.Scale(frame, from_=0.05, to=.95, variable=self.threshold, orient="horizontal").pack(fill="x")
        self.start_button = ttk.Button(frame, text="开始标注", style="Accent.TButton", command=self.start); self.start_button.pack(fill="x", pady=(16, 5))
        self.stop_button = ttk.Button(frame, text="停止任务", command=self.stop, state="disabled"); self.stop_button.pack(fill="x")
        self.progress = ttk.Progressbar(frame, mode="determinate"); self.progress.pack(fill="x", pady=(18, 5))
        self.progress_text = ttk.Label(frame, text="0 / 0", foreground="#718079"); self.progress_text.pack(anchor="e")

    def _build_center(self, frame):
        top = ttk.Frame(frame); top.pack(fill="x")
        ttk.Label(top, text="标注查看器", style="Section.TLabel").pack(side="left")
        self.image_name = ttk.Label(top, text="尚未加载图片", foreground="#718079"); self.image_name.pack(side="right")
        self.canvas = tk.Canvas(frame, background="#e6ebe8", highlightthickness=0); self.canvas.pack(fill="both", expand=True, pady=9)
        nav = ttk.Frame(frame); nav.pack(fill="x")
        self.prev_button = ttk.Button(nav, text="上一张", command=lambda: self.select_item(self.current - 1)); self.prev_button.pack(side="left")
        self.position = ttk.Label(nav, text="0 / 0"); self.position.pack(side="left", padx=12)
        self.next_button = ttk.Button(nav, text="下一张", command=lambda: self.select_item(self.current + 1)); self.next_button.pack(side="left")
        ttk.Button(nav, text="框视图", command=lambda: self.render("box")).pack(side="right")
        ttk.Button(nav, text="轮廓视图", command=lambda: self.render("mask")).pack(side="right", padx=(0, 6))
        self.canvas.bind("<ButtonPress-1>", self.begin_draw); self.canvas.bind("<B1-Motion>", self.drag_draw); self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.draw_mode = False; self.draw_box = None

    def _build_right(self, frame):
        ttk.Label(frame, text="复核与修改", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(frame, text="当前图片目标").pack(anchor="w")
        self.instance_list = tk.Listbox(frame, height=13, width=30, exportselection=False); self.instance_list.pack(fill="both", expand=True, pady=7)
        self.instance_list.bind("<<ListboxSelect>>", lambda e: self.render())
        ttk.Label(frame, text="补框/修改类别").pack(anchor="w")
        self.class_var = tk.StringVar(); self.class_combo = ttk.Combobox(frame, textvariable=self.class_var, state="readonly"); self.class_combo.pack(fill="x", pady=(4, 7)); self.class_combo.bind("<<ComboboxSelected>>", self.change_class)
        ttk.Button(frame, text="删除选中目标", command=self.delete_selected).pack(fill="x")
        ttk.Button(frame, text="保存当前修改", command=self.save_current).pack(fill="x", pady=5)
        self.review_var = tk.BooleanVar(); ttk.Checkbutton(frame, text="当前图片已复核", variable=self.review_var, command=self.save_current).pack(anchor="w", pady=(3, 16))
        ttk.Separator(frame).pack(fill="x", pady=(0, 14))
        ttk.Label(frame, text="导出说明", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(frame, text="结果目录会保留图片的相对路径：\nimages/原目录/图片.jpg\nlabels_detect/原目录/图片.txt\nlabels_segment/原目录/图片.txt", foreground="#718079", justify="left").pack(anchor="w")
        ttk.Button(frame, text="打开结果目录", command=self.open_output).pack(fill="x", pady=(15, 0))

    def choose_folder(self):
        path = filedialog.askdirectory(title="选择图片文件夹")
        if path: self._set_source(Path(path), False)

    def choose_zip(self):
        path = filedialog.askopenfilename(title="选择图集压缩包", filetypes=[("ZIP", "*.zip")])
        if path: self._set_source(Path(path), True)

    def _set_source(self, path, is_zip):
        try:
            files = discover_images(path, is_zip)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc)); return
        if not files: messagebox.showwarning("没有图片", "选择的来源里没有可读取的图片。"); return
        self.source, self.source_is_zip = path, is_zip; self.source_var.set(str(path)); self.status.config(text=f"找到 {len(files)} 张图片")
        if not self.output_var.get(): self.output_var.set(str(path.with_name(path.stem + "_labeled") if is_zip else path.parent / (path.name + "_labeled")))

    def choose_output(self):
        path = filedialog.askdirectory(title="选择标注结果目录")
        if path: self.output_var.set(path)

    def open_output(self):
        path = self.output or (Path(self.output_var.get()) if self.output_var.get() else None)
        if not path or not Path(path).exists():
            return messagebox.showinfo("结果目录", "标注完成后这里会打开结果目录。")
        path = str(Path(path).resolve())
        try:
            if hasattr(os, "startfile"):
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))

    def start(self):
        if not self.source: return messagebox.showwarning("缺少图集", "请先选择图片文件夹或 ZIP 压缩包。")
        if not self.output_var.get(): return messagebox.showwarning("缺少输出目录", "请选择结果目录。")
        try: classes, warnings = parse_prompts(self.prompt_text.get("1.0", "end"))
        except ValueError as exc: return messagebox.showerror("目标文本有误", str(exc))
        if Path(self.output_var.get()).resolve() == self.source.resolve() and not self.source_is_zip:
            return messagebox.showerror("输出目录无效", "输出目录不能与原图文件夹相同。")
        self.classes, self.items, self.current, self.stop_requested = classes, [], -1, False
        self.start_button.config(state="disabled"); self.stop_button.config(state="normal"); self.progress.config(value=0)
        threading.Thread(target=self._worker, args=(warnings,), daemon=True).start()

    def _worker(self, warnings):
        temp_dir = None
        try:
            source_root = self.source
            if self.source_is_zip:
                temp_dir = tempfile.TemporaryDirectory(prefix="sam3_zip_"); source_root = Path(temp_dir.name)
                with zipfile.ZipFile(self.source) as archive:
                    for info in archive.infolist():
                        target = (source_root / info.filename).resolve()
                        if info.is_dir() or not target.is_relative_to(source_root.resolve()) or Path(info.filename).suffix.lower() not in EXTENSIONS: continue
                        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(archive.read(info))
            files = discover_images(source_root)
            output = Path(self.output_var.get()).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
            self.output = output
            for index, relative in enumerate(files):
                if self.stop_requested: break
                source = source_root / relative; image_dir = output / "images" / relative.parent; label_dir = output / "labels_detect" / relative.parent; segment_dir = output / "labels_segment" / relative.parent
                image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True); segment_dir.mkdir(parents=True, exist_ok=True)
                out_image = image_dir / (relative.stem + ".jpg")
                with Image.open(source) as raw: image = ImageOps.exif_transpose(raw).convert("RGB"); width, height = image.size; image.save(out_image, quality=95)
                item = {"id": index, "name": str(relative), "file": str(out_image.relative_to(output)), "width": width, "height": height, "instances": [], "reviewed": False, "status": "running"}
                try:
                    from auto_label_pipeline import LabelPrompt, infer_instances
                    processor = self._model()
                    prompts = [LabelPrompt(c["id"], c["name"], c["prompts"]) for c in self.classes]; processor.confidence_threshold = float(self.threshold.get())
                    import torch
                    with Image.open(source) as source_image: source_image = ImageOps.exif_transpose(source_image).convert("RGB")
                    precision = torch.autocast("cuda", dtype=torch.bfloat16) if str(processor.device).startswith("cuda") else nullcontext()
                    with torch.inference_mode(), precision:
                        records = infer_instances(processor, source, prompts, 32, 2, 200, 1000, False)
                    item["instances"] = [{"class_id": r["class_id"], "score": r["score"], "box_xyxy": r["box_xyxy"], "polygon": r["polygon"], "manual": False} for r in records]
                    item["status"] = "done"
                    self._write_labels(output, relative, item["instances"], width, height)
                except Exception as exc: item.update(status="error", error=str(exc))
                self.items.append(item); self.queue.put(("progress", index + 1, len(files), item))
                self._save_manifest(output)
                gc.collect()
            self.queue.put(("done", warnings, len(files)))
        except Exception as exc: self.queue.put(("error", str(exc)))
        finally:
            if temp_dir: temp_dir.cleanup()

    def _model(self):
        if not hasattr(self, "processor") or self.processor is None:
            from auto_label_pipeline import build_model
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = ROOT / "checkpoints" / "sam3.pt"
            if not checkpoint.exists(): raise RuntimeError(f"未找到模型权重：{checkpoint}")
            self.processor = build_model(device, str(checkpoint), float(self.threshold.get()))
        return self.processor

    def _write_labels(self, output, relative, instances, width, height):
        detect = []; segment = []
        for item in instances:
            x1, y1, x2, y2 = item["box_xyxy"]; detect.append(f'{item["class_id"]} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
            segment.append(str(item["class_id"]) + " " + " ".join(f"{v:.6f}" for v in item["polygon"]))
        (output / "labels_detect" / relative.parent / (relative.stem + ".txt")).write_text("\n".join(detect), encoding="utf-8")
        (output / "labels_segment" / relative.parent / (relative.stem + ".txt")).write_text("\n".join(segment), encoding="utf-8")

    def _save_manifest(self, output):
        data = {"classes": self.classes, "items": self.items, "source": str(self.source), "updated": True}
        (output / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "data.yaml").write_text(yaml.safe_dump({"path": str(output), "train": "images", "names": {c["id"]: c["name"] for c in self.classes}}, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _drain_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total, item = msg; self.progress.config(maximum=total, value=done); self.progress_text.config(text=f"{done} / {total}"); self.status.config(text=f"{item['name']}：{item['status']}"); self.items = self.items
                    if self.current < 0:
                        self.current = 0
                        self.image_name.config(text=self.items[0]["name"])
                        self.position.config(text=f"1 / {len(self.items)}")
                    self.refresh_list()
                elif msg[0] == "done":
                    self.start_button.config(state="normal"); self.stop_button.config(state="disabled"); self.status.config(text="标注完成"); self.refresh_list();
                    if msg[1]: messagebox.showwarning("提示", "\n".join(msg[1]))
                elif msg[0] == "error":
                    self.start_button.config(state="normal"); self.stop_button.config(state="disabled"); messagebox.showerror("标注失败", msg[1])
        except queue.Empty: pass
        self.root.after(120, self._drain_queue)

    def stop(self): self.stop_requested = True; self.status.config(text="将在当前图片完成后停止…")
    def refresh_list(self):
        self.instance_list.delete(0, "end")
        self.class_combo["values"] = [c["name"] for c in self.classes]
        if self.classes and not self.class_var.get(): self.class_var.set(self.classes[0]["name"])
        if 0 <= self.current < len(self.items):
            item = self.items[self.current]; self.review_var.set(item.get("reviewed", False))
            for i in item.get("instances", []): self.instance_list.insert("end", f'{self.classes[i["class_id"]]["name"]}  {i["score"]:.0%}')
        self.render()

    def select_item(self, index):
        if self.dirty: self.save_current()
        if 0 <= index < len(self.items): self.current = index; self.dirty = False; self.image_name.config(text=self.items[index]["name"]); self.position.config(text=f"{index+1} / {len(self.items)}"); self.refresh_list()

    def render(self, mode="box"):
        if not 0 <= self.current < len(self.items): return
        item = self.items[self.current]; path = self.output / item["file"]
        try: self.image = Image.open(path).convert("RGB")
        except Exception: return
        self.canvas.delete("all"); cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200); scale = min(cw/self.image.width, ch/self.image.height); w, h = int(self.image.width*scale), int(self.image.height*scale); self.tk_image = ImageTk.PhotoImage(self.image.resize((w,h))); x, y = (cw-w)//2, (ch-h)//2; self.canvas.create_image(x,y,image=self.tk_image,anchor="nw")
        self.render_origin = (x,y,w,h)
        for n, r in enumerate(item.get("instances", [])):
            col = COLORS[r["class_id"] % len(COLORS)]; x1,y1,x2,y2 = [v*scale for v in r["box_xyxy"]]; x1+=x;y1+=y;x2+=x;y2+=y
            if mode == "mask":
                points=[]
                for p in range(0,len(r["polygon"]),2): points.extend((x+r["polygon"][p]*w,y+r["polygon"][p+1]*h))
                self.canvas.create_polygon(points, outline=col, fill=col, stipple="gray25", width=2)
            else: self.canvas.create_rectangle(x1,y1,x2,y2,outline=col,width=2)
            self.canvas.create_text(x1+3,y1+3,anchor="nw",text=self.classes[r["class_id"]]["name"],fill=col,font=("TkDefaultFont",10,"bold"))

    def delete_selected(self):
        if 0 <= self.current < len(self.items) and self.instance_list.curselection():
            del self.items[self.current]["instances"][self.instance_list.curselection()[0]]; self.save_current()

    def change_class(self, _event=None):
        if not 0 <= self.current < len(self.items) or not self.instance_list.curselection(): return
        selected_name = self.class_var.get()
        class_id = next((c["id"] for c in self.classes if c["name"] == selected_name), None)
        if class_id is None: return
        self.items[self.current]["instances"][self.instance_list.curselection()[0]]["class_id"] = class_id
        self.save_current()

    def save_current(self):
        if not 0 <= self.current < len(self.items): return
        item = self.items[self.current]; item["reviewed"] = self.review_var.get(); relative = Path(item["name"]); self._write_labels(self.output, relative, item["instances"], item["width"], item["height"]); self._save_manifest(self.output); self.refresh_list()

    def begin_draw(self, event):
        if not 0 <= self.current < len(self.items): return
        self.draw_box = (event.x, event.y, event.x, event.y)
    def drag_draw(self, event):
        if self.draw_box:
            self.draw_box = (self.draw_box[0], self.draw_box[1], event.x, event.y); self.render(); self.canvas.create_rectangle(*self.draw_box, outline="white", dash=(4,3), width=2)
    def end_draw(self, event):
        if not self.draw_box: return
        x0,y0,x1,y1=self.draw_box; self.draw_box=None
        ox,oy,w,h=self.render_origin; ax,ay,bx,by=min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)
        if bx-ax > 5 and by-ay > 5:
            source = next((c["id"] for c in self.classes if c["name"] == self.class_var.get()), 0); image=self.items[self.current]; sx=image["width"]/w; sy=image["height"]/h; px1,py1,px2,py2=(ax-ox)*sx,(ay-oy)*sy,(bx-ox)*sx,(by-oy)*sy; image["instances"].append({"class_id":source,"score":1,"manual":True,"box_xyxy":[px1,py1,px2,py2],"polygon":[px1/image["width"],py1/image["height"],px2/image["width"],py1/image["height"],px2/image["width"],py2/image["height"],px1/image["width"],py2/image["height"]]}); self.save_current()


def main():
    root = tk.Tk(); DesktopApp(root); root.mainloop()


if __name__ == "__main__": main()
