"""Local text-to-YOLO workspace. Model imports are deferred until a job starts."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import threading
import tempfile
import uuid
import zipfile

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALIASES = {
    "人": "person", "行人": "person", "人脸": "face", "手": "hand",
    "拳头": "fist", "猫": "cat", "狗": "dog", "汽车": "car",
    "车": "car", "自行车": "bicycle", "摩托车": "motorcycle",
    "卡车": "truck", "公交车": "bus", "瓶子": "bottle", "杯子": "cup",
    "椅子": "chair", "桌子": "table", "手机": "mobile phone",
    "电脑": "computer", "安全帽": "helmet", "红色汽车": "red car",
    "交通灯": "traffic light", "刀": "knife", "剑": "sword", "矛": "spear",
}


def parse_prompts(text):
    classes = []
    warnings = []
    if not isinstance(text, str) or len(text) > 8000:
        raise ValueError("目标文本过长或无效。")
    for line in text.replace("；", "\n").replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        name, separator, prompt = line.partition("=")
        name = name.strip()
        phrases = [p.strip() for p in (prompt if separator else name).split("|")]
        phrases = list(dict.fromkeys(ALIASES.get(p, p) for p in phrases))
        if not name or len(name) > 100 or any(not p or len(p) > 300 for p in phrases):
            raise ValueError("每行填写一个目标，格式为 类别名 = 描述；描述不能留空。")
        if name in [c["name"] for c in classes]:
            raise ValueError(f"类别名重复：{name}")
        if any(any("\u4e00" <= ch <= "\u9fff" for ch in p) for p in phrases):
            warnings.append(f"“{name}”含未翻译的中文描述，建议填写英文描述以提高识别率。")
        classes.append({"id": len(classes), "name": name, "prompts": phrases})
    if not classes or len(classes) > 40:
        raise ValueError("请输入 1 到 40 个类别，每行一个。")
    return classes, warnings


def validate_instances(instances, classes, width, height):
    if not isinstance(instances, list) or len(instances) > 1000:
        raise ValueError("标注列表无效。")
    cleaned = []
    for item in instances:
        cid = item.get("class_id")
        if type(cid) is not int or not 0 <= cid < len(classes):
            raise ValueError("类别编号无效。")
        box = item.get("box_xyxy", [])
        if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
            raise ValueError("检测框坐标无效。")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("检测框必须在图片内且有正面积。")
        polygon = item.get("polygon", [])
        if len(polygon) < 6 or len(polygon) % 2 or len(polygon) > 4000:
            raise ValueError("分割轮廓至少需要三个点。")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in polygon):
            raise ValueError("分割坐标必须在 0 到 1 之间。")
        score = float(item.get("score", 1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("置信度无效。")
        cleaned.append({"class_id": cid, "box_xyxy": box, "polygon": polygon,
                        "score": score, "manual": bool(item.get("manual", False))})
    return cleaned


class Workspace:
    def __init__(self, directory, checkpoint, device="auto"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.processor = None
        self.jobs = {}
        for file in self.directory.glob("*/job.json"):
            try:
                job = json.loads(file.read_text(encoding="utf-8"))
                if job["status"] in {"queued", "running", "cancelling"}:
                    job.update(status="interrupted", message="上次运行已中断，已完成的标注仍可查看和导出。")
                self.jobs[job["id"]] = job
            except (ValueError, KeyError):
                continue

    def save(self, job):
        path = self.directory / job["id"] / "job.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def get(self, job_id):
        if job_id not in self.jobs:
            abort(404)
        return self.jobs[job_id]

    def model(self):
        if self.processor is None:
            if not self.checkpoint.is_file():
                raise RuntimeError(f"未找到 SAM3 权重：{self.checkpoint}")
            import torch
            from auto_label_pipeline import build_model
            device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            self.processor = build_model(device, str(self.checkpoint), 0.35)
        return self.processor

    def run(self, job_id):
        job = self.jobs[job_id]
        with self.inference_lock:
            try:
                with self.lock:
                    if job["status"] == "cancelling":
                        job.update(status="cancelled", message="任务已取消。")
                        self.save(job)
                        return
                    job.update(status="running", message="正在加载 SAM3 模型…")
                    self.save(job)
                processor = self.model()
                import torch
                from auto_label_pipeline import LabelPrompt, infer_instances
                prompts = [LabelPrompt(c["id"], c["name"], c["prompts"]) for c in job["classes"]]
                processor.confidence_threshold = job["threshold"]
                for item in job["items"]:
                    with self.lock:
                        if job["status"] == "cancelling":
                            break
                        job["message"] = f"正在标注 {item['name']}"
                    path = self.directory / job_id / "images" / item["file"]
                    try:
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if str(processor.device).startswith("cuda") else nullcontext()
                        with torch.inference_mode(), precision:
                            records = infer_instances(processor, path, prompts, 32, 2, 200, 1000, False)
                        instances = []
                        for record in records:
                            box = record["box_xyxy"]
                            box = [max(0., min(float(v), item["width"] if i % 2 == 0 else item["height"])) for i, v in enumerate(box)]
                            if box[2] <= box[0] or box[3] <= box[1]:
                                continue
                            instances.append({"class_id": record["class_id"], "score": record["score"],
                                              "box_xyxy": box, "polygon": record["polygon"], "manual": False})
                        with self.lock:
                            item.update(status="done", instances=instances)
                    except torch.cuda.OutOfMemoryError:
                        raise RuntimeError("显存不足。请关闭其他 GPU 程序，或使用 --device cpu 启动后重试。")
                    except Exception as exc:
                        with self.lock:
                            item.update(status="error", error=str(exc))
                    with self.lock:
                        job["completed"] += 1
                        self.save(job)
                    gc.collect()
                with self.lock:
                    cancelled = job["status"] == "cancelling"
                    errors = sum(i["status"] == "error" for i in job["items"])
                    job.update(status="cancelled" if cancelled else "done", message="任务已取消，已完成结果已保存。" if cancelled else f"标注完成，{errors} 张失败。")
            except Exception as exc:
                with self.lock:
                    job.update(status="error", message=str(exc))
            finally:
                with self.lock:
                    self.save(job)
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass


def create_app(workspace=None):
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["MAX_FORM_PARTS"] = 5000
    ws = workspace or Workspace(ROOT / "runs" / "studio", ROOT / "checkpoints" / "sam3.pt")
    app.extensions["workspace"] = ws

    @app.before_request
    def local_only():
        if request.host.split(":")[0] not in {"localhost", "127.0.0.1"}:
            abort(403)
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403)

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(exc):
        return jsonify(error="上传超过 512MB，请使用本地文件夹导入。"), 413

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/info")
    def info():
        return jsonify(checkpoint=ws.checkpoint.is_file(), default_folder=str(ROOT / "src" / "images"), device=ws.device)

    @app.post("/api/prompts")
    def prompts():
        classes, warnings = parse_prompts(request.get_json()["text"])
        return jsonify(classes=classes, warnings=warnings)

    @app.get("/api/jobs")
    def jobs():
        with ws.lock:
            return jsonify([{k: j[k] for k in ("id", "created", "status", "completed", "total", "text", "message")} for j in reversed(list(ws.jobs.values()))])

    @app.post("/api/jobs")
    def start():
        classes, warnings = parse_prompts(request.form.get("text", ""))
        threshold = float(request.form.get("threshold", ".35"))
        if not math.isfinite(threshold) or not 0.01 <= threshold <= 0.99:
            raise ValueError("置信度应在 0.01 到 0.99 之间。")
        with ws.lock:
            if any(j["status"] in {"queued", "running", "cancelling"} for j in ws.jobs.values()):
                return jsonify(error="已有任务在运行，请等待或取消后再开始。"), 409
            job_id = uuid.uuid4().hex
            directory = ws.directory / job_id / "images"
            directory.mkdir(parents=True)
            items = []
            rejected = []

            def add_image(source, name):
                try:
                    with Image.open(source) as raw:
                        rgb = ImageOps.exif_transpose(raw).convert("RGB")
                        if rgb.width * rgb.height > 40_000_000:
                            raise ValueError("图片超过 4000 万像素")
                        file = f"{len(items):06d}.jpg"
                        rgb.save(directory / file, quality=95)
                        items.append({"id": len(items), "name": name, "file": file,
                                      "width": rgb.width, "height": rgb.height,
                                      "status": "pending", "instances": [], "reviewed": False})
                except Exception as exc:
                    rejected.append(f"{name}: {exc}")

            folder = request.form.get("folder", "").strip()
            limit = int(request.form.get("limit", "0"))
            if not 0 <= limit <= 10000:
                raise ValueError("图片上限应在 0 到 10000 之间，0 表示全部。")
            if folder:
                source_dir = Path(folder).expanduser().resolve()
                if not source_dir.is_dir():
                    raise ValueError("本地图片文件夹不存在。")
                paths = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.is_relative_to(ws.directory.resolve()))
                if limit:
                    paths = paths[:limit]
                if len(paths) > 10000:
                    raise ValueError("单次最多导入 10000 张图片，请设置图片上限。")
                for path in paths:
                    add_image(path, str(path.relative_to(source_dir)))
            else:
                uploads = request.files.getlist("images")
                for upload in uploads[:limit or None]:
                    if Path(upload.filename or "").suffix.lower() in EXTENSIONS:
                        add_image(upload.stream, Path(upload.filename).name)
            if not items:
                raise ValueError("没有可读取的图片。" + "；".join(rejected[:3]))
            job = {"id": job_id, "created": datetime.now().isoformat(timespec="seconds"), "text": request.form["text"],
                   "classes": classes, "warnings": warnings, "rejected": rejected, "threshold": threshold,
                   "status": "queued", "message": "等待模型加载…", "completed": 0, "total": len(items), "items": items}
            ws.jobs[job_id] = job
            ws.save(job)
            threading.Thread(target=ws.run, args=(job_id,), daemon=True).start()
        return jsonify(id=job_id), 201

    @app.get("/api/jobs/<job_id>")
    def job(job_id):
        with ws.lock:
            return jsonify(deepcopy(ws.get(job_id)))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel(job_id):
        with ws.lock:
            job = ws.get(job_id)
            if job["status"] in {"queued", "running"}:
                job.update(status="cancelling", message="将在当前图片处理完成后停止。")
                ws.save(job)
        return jsonify(ok=True)

    def find_item(job_id, item_id):
        job = ws.get(job_id)
        if not 0 <= item_id < len(job["items"]):
            abort(404)
        return job, job["items"][item_id]

    @app.get("/api/jobs/<job_id>/images/<int:item_id>")
    def picture(job_id, item_id):
        with ws.lock:
            _, item = find_item(job_id, item_id)
            path = ws.directory / job_id / "images" / item["file"]
        return send_file(path)

    @app.put("/api/jobs/<job_id>/items/<int:item_id>")
    def edit(job_id, item_id):
        with ws.lock:
            job, item = find_item(job_id, item_id)
            if item["status"] != "done":
                return jsonify(error="只能修改已完成的图片。"), 409
            data = request.get_json()
            item["instances"] = validate_instances(data["instances"], job["classes"], item["width"], item["height"])
            item["reviewed"] = bool(data.get("reviewed", False))
            ws.save(job)
        return jsonify(ok=True)

    @app.get("/api/jobs/<job_id>/export")
    def export(job_id):
        with ws.lock:
            job = deepcopy(ws.get(job_id))
        mode = request.args.get("format", "detect")
        if mode not in {"detect", "segment"}:
            raise ValueError("导出格式无效。")
        selected = [i for i in job["items"] if i["status"] == "done" and (request.args.get("reviewed") != "1" or i["reviewed"])]
        if not selected:
            raise ValueError("没有符合条件的已完成图片。")
        shuffled = list(selected)
        random.Random(42).shuffle(shuffled)
        val_ids = {i["id"] for i in shuffled[:max(1, round(len(shuffled) * .2))]} if len(shuffled) > 1 else set()
        # Large datasets spill to disk instead of holding the whole ZIP in RAM.
        buffer = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024)
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in selected:
                split = "val" if item["id"] in val_ids else "train"
                archive.write(ws.directory / job_id / "images" / item["file"], f"images/{split}/{item['file']}")
                lines = []
                for instance in item["instances"]:
                    if mode == "segment":
                        coords = instance["polygon"]
                    else:
                        x1, y1, x2, y2 = instance["box_xyxy"]
                        w, h = item["width"], item["height"]
                        coords = [(x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h]
                    lines.append(str(instance["class_id"]) + " " + " ".join(f"{c:.6f}" for c in coords))
                archive.writestr(f"labels/{split}/{Path(item['file']).stem}.txt", "\n".join(lines))
            config = {"train": "images/train", "val": "images/val" if val_ids else None,
                      "names": {c["id"]: c["name"] for c in job["classes"]}}
            archive.writestr("data.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            archive.writestr("manifest.json", json.dumps({**job, "items": selected, "export_format": mode}, ensure_ascii=False, indent=2))
            archive.writestr("README.txt", "YOLO " + mode + "\nImages are EXIF-normalized. Empty labels are negative samples.\nTrain/val split: seeded random 80/20. For video or correlated images, split by source before training.\nA single-image export has no validation set; add validation images before training.\nManual boxes use rectangular polygons in segmentation exports.\n")
        buffer.seek(0)
        response = send_file(buffer, mimetype="application/zip", as_attachment=True, download_name=f"yolo-{mode}-{job_id[:8]}.zip")
        response.call_on_close(buffer.close)
        return response

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 text annotation studio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "sam3.pt"))
    parser.add_argument("--workspace", default=str(ROOT / "runs" / "studio"))
    args = parser.parse_args()
    create_app(Workspace(args.workspace, args.checkpoint, args.device)).run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
