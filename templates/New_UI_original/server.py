#!/usr/bin/env python3
import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


ROOT = os.path.dirname(os.path.abspath(__file__))


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


class Handler(SimpleHTTPRequestHandler):
    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def _read_body(self):
        length = int(self.headers.get('Content-Length', '0') or 0)
        if length:
            return self.rfile.read(length)
        return b''

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/health':
            self._set_json_headers(200)
            self.wfile.write(json_bytes({"status": "ok"}))
            return

        if path == '/inheritance_catalog':
            payload = {
                "available_years": [2020, 2021, 2022, 2023, 2024, 2025],
                "authors": [
                    "陳小明 (National Taiwan University)",
                    "林大華 (NCKU)",
                    "王美玲 (NTUST)",
                    "張家豪 (Academia Sinica)"
                ],
            }
            self._set_json_headers(200)
            self.wfile.write(json_bytes(payload))
            return

        if path == '/categories_data':
            papers = [
                {
                    "論文標題": "Remote Sensing in 2024: A Survey",
                    "研究目的": ["分類", "變遷偵測"],
                    "年份": ["2024"],
                    "資料集": ["UC Merced", "BigEarthNet"],
                    "建模": ["ResNet50", "Swin Transformer"],
                    "資料前處理": ["影像正規化", "資料增強"],
                    "評估指標": ["Accuracy", "F1 Score"],
                },
                {
                    "論文標題": "Coastal Erosion Mapping with SAR",
                    "研究目的": ["變遷偵測"],
                    "年份": ["2023"],
                    "資料集": ["Sentinel-1", "Landsat 8"],
                    "建模": ["U-Net", "UNet++"],
                    "資料前處理": ["Speckle 降噪", "配準"],
                    "評估指標": ["IoU", "Dice"],
                },
                {
                    "論文標題": "Land Use Classification with Foundation Models",
                    "研究目的": ["分類"],
                    "年份": ["2025"],
                    "資料集": ["DeepGlobe", "xView"],
                    "建模": ["CLIP", "ViT"],
                    "資料前處理": ["切片", "標註清理"],
                    "評估指標": ["Top-1", "Top-5"],
                },
                {
                    "論文標題": "Mangrove Detection Using Multispectral Data",
                    "研究目的": ["偵測", "分類"],
                    "年份": ["2022", "2023"],
                    "資料集": ["PlanetScope", "WorldView-2"],
                    "建模": ["YOLOv5", "EfficientNet"],
                    "資料前處理": ["大氣校正"],
                    "評估指標": ["mAP", "Precision", "Recall"],
                },
                {
                    "論文標題": "Change Detection Benchmark for Taiwan",
                    "研究目的": ["變遷偵測"],
                    "年份": ["2021", "2022"],
                    "資料集": ["Formosat-2", "Landsat 7"],
                    "建模": ["Siamese U-Net"],
                    "資料前處理": ["配準", "雲遮罩移除"],
                    "評估指標": ["F1 Score", "Kappa"],
                },
                {
                    "論文標題": "Road Extraction from Aerial Imagery",
                    "研究目的": ["偵測"],
                    "年份": ["2020", "2021"],
                    "資料集": ["Massachusetts Roads", "DeepGlobe"],
                    "建模": ["DeepLabv3+", "HRNet"],
                    "資料前處理": ["直方圖等化"],
                    "評估指標": ["IoU", "F1 Score"],
                },
            ]
            self._set_json_headers(200)
            self.wfile.write(json_bytes({"papers": papers}))
            return

        # Serve static files for other paths
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()
        ctype = (self.headers.get('Content-Type') or '').lower()

        def try_parse_json():
            try:
                return json.loads(body.decode('utf-8')) if body else {}
            except Exception:
                return {}

        if path == '/query':
            _ = try_parse_json()
            self._set_json_headers(200)
            self.wfile.write(json_bytes({
                "action": "mock_web_search",
                "output": "這是示範回覆：已接收你的查詢，這裡顯示模擬結果與摘要。",
            }))
            return

        if path == '/inheritance_query':
            _ = try_parse_json()
            self._set_json_headers(200)
            self.wfile.write(json_bytes({
                "output": "示範：已分析研究者脈絡與學術系譜（mock）。",
            }))
            return

        if path == '/upload_paper':
            # Frontend sends multipart/form-data; we don't need to parse for demo
            self._set_json_headers(200)
            payload = {
                "success": True,
                "stored_filename": "demo_paper.pdf",
                "paper_data": {
                    "論文標題": "Demo Paper Title",
                    "年份": ["2024"],
                    "作者": ["測試作者A", "測試作者B"],
                    "研究目的": ["分類", "偵測"],
                    "資料集": ["UC Merced", "DeepGlobe"],
                    "資料前處理": ["正規化", "資料增強"],
                    "建模": ["ResNet50", "U-Net"],
                    "評估指標": ["Accuracy", "F1 Score"],
                    "其他": ["附註：此為示範資料"]
                },
                "classification": {
                    "area_name": "Remote Sensing",
                    "trunk_name": "Land Use / Land Cover",
                    "reasoning": "依據標題與關鍵字判定（demo）。"
                }
            }
            self.wfile.write(json_bytes(payload))
            return

        # Fallback
        self._set_json_headers(404)
        self.wfile.write(json_bytes({"error": "Not Found"}))


def run(addr='127.0.0.1', port=8000):
    os.chdir(ROOT)
    httpd = HTTPServer((addr, port), Handler)
    print(f"Demo server running at http://{addr}:{port}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    run(host, port)

