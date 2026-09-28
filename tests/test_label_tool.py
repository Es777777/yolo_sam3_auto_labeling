import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_tool import Workspace, create_app, parse_prompts, validate_instances


def test_prompts_preserve_class_ids_and_translate_known_words():
    classes, warnings = parse_prompts("汽车\n帽子 = safety helmet | hard hat\nmy object")
    assert [c["id"] for c in classes] == [0, 1, 2]
    assert classes[0]["prompts"] == ["car"]
    assert classes[1]["prompts"] == ["safety helmet", "hard hat"]
    assert not warnings
    assert parse_prompts("红色的工业夹具")[1]
    for invalid in ("", "a\na", "x = ", "x = a | "):
        with pytest.raises(ValueError):
            parse_prompts(invalid)


@pytest.fixture
def setup(tmp_path):
    ws = Workspace(tmp_path / "workspace", tmp_path / "sam3.pt")
    images = ws.directory / "example" / "images"
    images.mkdir(parents=True)
    instance = {"class_id": 0, "box_xyxy": [10, 20, 50, 60], "polygon": [.1,.2,.5,.2,.5,.6,.1,.6], "score": .8}
    items = []
    for index in range(4):
        file = f"{index:06d}.jpg"
        Image.new("RGB", (100,100), "white").save(images / file)
        items.append({"id":index,"name":f"source{index}.png","file":file,"width":100,"height":100,
                      "status":"done" if index < 3 else "error", "instances":[instance] if index == 0 else [], "reviewed":index == 0})
    job = {"id":"example","created":"2026-01-01","status":"done","message":"done","text":"car",
           "classes":[{"id":0,"name":"car","prompts":["car"]}], "items":items,"total":4,"completed":4}
    ws.jobs[job["id"]] = job
    ws.save(job)
    app = create_app(ws)
    app.config["TESTING"] = True
    return ws, app.test_client(), instance


def test_export_split_coordinates_negatives_and_errors(setup):
    ws, client, _ = setup
    response = client.get("/api/jobs/example/export?format=detect")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        names = archive.namelist()
        assert len([p for p in names if p.endswith(".jpg")]) == 3
        assert not any("000003" in p for p in names)
        train = [p for p in names if p.startswith("images/train/")]
        val = [p for p in names if p.startswith("images/val/")]
        assert len(train) == 2 and len(val) == 1
        label = next(p for p in names if p.endswith("000000.txt"))
        assert archive.read(label).decode() == "0 0.300000 0.400000 0.400000 0.400000"
        negative = next(p for p in names if p.endswith("000001.txt"))
        assert archive.read(negative) == b""
        config = yaml.safe_load(archive.read("data.yaml"))
        assert "path" not in config and config["names"] == {0:"car"}
        assert len(json.loads(archive.read("manifest.json"))["items"]) == 3
    response = client.get("/api/jobs/example/export?format=segment&reviewed=1")
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        label = archive.read("labels/train/000000.txt").decode().split()
        assert len(label) == 9
        assert yaml.safe_load(archive.read("data.yaml"))["val"] is None


def test_edits_persist_and_reject_invalid_coordinates(setup):
    ws, client, instance = setup
    url = "/api/jobs/example/items/0"
    for changes in ({"class_id":9}, {"box_xyxy":[0,0,101,40]}, {"polygon":[0,0,float('nan'),0,1,1]}):
        response = client.put(url, json={"instances":[{**instance,**changes}]})
        assert response.status_code == 400
    assert client.put(url,json={"instances":[],"reviewed":True}).status_code == 200
    restored = Workspace(ws.directory,ws.checkpoint)
    assert restored.jobs["example"]["items"][0]["instances"] == []
    assert restored.jobs["example"]["items"][0]["reviewed"] is True
    assert client.put("/api/jobs/example/items/3",json={"instances":[]}).status_code == 409


def test_import_same_names_unique_ids_and_exif_orientation(setup, tmp_path, monkeypatch):
    ws, client, _ = setup
    monkeypatch.setattr(ws,"run",lambda job_id: None)
    folder = tmp_path / "source"
    for sub in ["a","b"]:
        (folder / sub).mkdir(parents=True)
        im = Image.new("RGB",(30,50))
        exif = im.getexif()
        exif[274] = 6
        im.save(folder / sub / "same.jpg", exif=exif)
    response = client.post("/api/jobs",data={"text":"car", "folder":str(folder)})
    assert response.status_code == 201
    job = ws.jobs[response.json["id"]]
    assert len({i["file"] for i in job["items"]}) == 2
    assert job["items"][0]["width"] == 50 and job["items"][0]["height"] == 30
    assert client.post("/api/jobs",data={"text":"car","folder":str(folder)}).status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    restored = Workspace(ws.directory, ws.checkpoint)
    assert restored.jobs[job["id"]]["status"] == "interrupted"


def test_reject_cross_origin_and_external_host(setup):
    _, client, _ = setup
    assert client.post("/api/prompts",json={"text":"car"},headers={"Origin":"https://example.com"}).status_code == 403
    assert client.get("/api/info",headers={"Host":"example.com"}).status_code == 403
