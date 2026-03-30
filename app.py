import os
import json
import uuid
import traceback
from datetime import datetime

import cv2
import torch
import timm
import gspread
import numpy as np

from flask import Flask, request, send_from_directory
from oauth2client.service_account import ServiceAccountCredentials
from torchvision import transforms
from ultralytics import YOLO

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    ImageMessage,
    TextSendMessage,
    ImageSendMessage
)

# =========================================================
# Reduce crash risk on small containers
# =========================================================
cv2.setNumThreads(0)
torch.set_num_threads(1)

# =========================================================
# CONFIG FROM ENVIRONMENT VARIABLES
# =========================================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

SHEET_KEY = os.environ.get("SHEET_KEY", "")
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "Dashboard")
SERVICE_ACCOUNT_JSON_PATH = os.environ.get("SERVICE_ACCOUNT_JSON_PATH", "")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
CLASS_MAP_PATH = os.environ.get("CLASS_MAP_PATH", "models/class_mapping.json")

IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.20))
MIN_CONF = float(os.environ.get("MIN_CONF", 0.70))
GLOBAL_CONF = float(os.environ.get("GLOBAL_CONF", 0.70))

INPUT_DIR = os.environ.get("INPUT_DIR", "inputs")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")

MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", 960))
MAX_BOXES = int(os.environ.get("MAX_BOXES", 3))

# =========================================================
# BASIC SETUP
# =========================================================
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

app = Flask(__name__)

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("Missing LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# =========================================================
# GOOGLE SHEETS
# =========================================================
sheet = None


def init_google_sheet():
    global sheet

    if not SERVICE_ACCOUNT_JSON_PATH:
        print("⚠️ SERVICE_ACCOUNT_JSON_PATH is empty. Google Sheets logging disabled.")
        return

    if not SHEET_KEY:
        print("⚠️ SHEET_KEY is empty. Google Sheets logging disabled.")
        return

    if not os.path.exists(SERVICE_ACCOUNT_JSON_PATH):
        print(f"⚠️ Service account file not found: {SERVICE_ACCOUNT_JSON_PATH}")
        return

    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_name(
            SERVICE_ACCOUNT_JSON_PATH,
            scope
        )
        gs_client = gspread.authorize(creds)
        sheet = gs_client.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
        print("✅ Google Sheets connected successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Google Sheets: {e}")
        traceback.print_exc()
        sheet = None


def log_to_sheet(text: str):
    global sheet

    if sheet is None:
        print(f"⚠️ Sheet not available. Skip logging: {text}")
        return

    try:
        now = datetime.now().strftime("%d/%m/%Y")
        row = [""] * 12 + [now, text]
        last_row = len(sheet.get_all_values()) + 1
        sheet.insert_row(row, last_row)
        print(f"✅ log_to_sheet: {text} (row {last_row})")
    except Exception as e:
        print(f"❌ log_to_sheet error: {e}")
        traceback.print_exc()


# =========================================================
# CLASS MAP
# =========================================================
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError(f"Class map file not found: {CLASS_MAP_PATH}")

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)

class_names = [class_map[str(i)] for i in range(len(class_map))]
NUM_CLASSES = len(class_names)

print("✅ โหลด class_mapping สำเร็จ")
print("NUM_CLASSES =", NUM_CLASSES)
print("class_names =", class_names)


def clean_class_name(name: str) -> str:
    name = name.replace("Tomato__Tomato_", "Tomato_")
    name = name.replace("Tomato_Tomato_", "Tomato_")
    name = name.replace("Tomato__Target_Spot", "Tomato_Target_Spot")

    while "Tomato__Tomato_" in name:
        name = name.replace("Tomato__Tomato_", "Tomato_")

    while "Tomato_Tomato_" in name:
        name = name.replace("Tomato_Tomato_", "Tomato_")

    return name


def display_class_name(name: str) -> str:
    clean = clean_class_name(name)
    clean = clean.replace("Tomato__", "")
    clean = clean.replace("Tomato_", "")
    return clean


def is_not_leaf(name: str) -> bool:
    return "not_a_leaf" in name.lower()


def is_healthy(name: str) -> bool:
    return "healthy" in name.lower()


# =========================================================
# DISEASE INFO
# =========================================================
disease_info = {
    "Tomato_Bacterial_spot": "🍂 โรคใบจุดแบคทีเรีย\nหลีกเลี่ยงน้ำกระเด็น ใช้เมล็ดพันธุ์ปลอดโรค และพิจารณาสารกลุ่มคอปเปอร์",
    "Tomato_Early_blight": "🍁 โรคใบจุดวง\nตัดใบที่เป็นโรค ลดความชื้นสะสม และพ่นสารป้องกันเชื้อราเมื่อจำเป็น",
    "Tomato_Late_blight": "🌧️ โรคใบไหม้\nควรกำจัดส่วนที่ติดเชื้อโดยเร็ว และจัดการความชื้นในแปลง",
    "Tomato_Leaf_Mold": "🍃 โรคราใบสีน้ำตาล\nเพิ่มการระบายอากาศ ลดความชื้น และเฝ้าระวังการระบาดต่อเนื่อง",
    "Tomato_Septoria_leaf_spot": "⚫ โรคใบจุดเซพโทเรีย\nตัดใบที่เป็นโรค ทำความสะอาดเศษพืช และพ่นสารป้องกันเชื้อราเมื่อเหมาะสม",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ ไรแดง\nตรวจใต้ใบ ฉีดน้ำแรงพอเหมาะใต้ใบ หรือใช้สารกำจัดไรตามความเหมาะสม",
    "Tomato_Target_Spot": "🎯 โรคใบจุดเป้ากระสุน\nหลีกเลี่ยงน้ำขัง ลดความชื้น และพ่นสารป้องกันเชื้อราเมื่อจำเป็น",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "🌀 โรคไวรัสใบหงิกเหลืองมะเขือเทศ\nกำจัดแมลงพาหะ เช่น แมลงหวี่ขาว และถอนต้นที่ติดเชื้อรุนแรง",
    "Tomato_Yellow_Leaf_Curl_Virus": "🌀 โรคไวรัสใบหงิกเหลืองมะเขือเทศ\nกำจัดแมลงพาหะ เช่น แมลงหวี่ขาว และถอนต้นที่ติดเชื้อรุนแรง",
    "Tomato_healthy": "✅ ต้นมะเขือเทศดูปกติดี",
    "not_a_leaf": "📷 ภาพนี้ไม่ใช่ใบมะเขือเทศหรือไม่ชัดพอ"
}


def get_disease_detail(class_name: str) -> str:
    class_name = clean_class_name(class_name)
    return disease_info.get(class_name, "ℹ️ ไม่มีข้อมูลเพิ่มเติม")


# =========================================================
# LOAD PYTORCH MODEL
# =========================================================
print("🚀 กำลังโหลดโมเดล PyTorch...")

model = timm.create_model(
    "mobilenetv4_conv_small",
    pretrained=False,
    num_classes=NUM_CLASSES
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("MODEL_PATH =", MODEL_PATH)
print("MODEL_EXISTS =", os.path.exists(MODEL_PATH))
print("MODEL_SIZE_BYTES =", os.path.getsize(MODEL_PATH))

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(checkpoint, dict):
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✅ โหลด best_model.pth สำเร็จ")


# =========================================================
# LOAD YOLO
# =========================================================
print("🚀 กำลังโหลด YOLO...")
yolo_model = YOLO("yolov8n.pt")
print("✅ โหลด YOLO สำเร็จ")


# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def resize_if_needed(img, max_side=960):
    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side <= max_side:
        return img

    scale = max_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def classify_leaf(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()


def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    box_y1 = max(0, y1 - th - 12)
    box_y2 = y1
    box_x2 = min(img.shape[1], x1 + tw + 12)

    cv2.rectangle(img, (x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 5, max(15, y1 - 6)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def log_multiple_diseases(detections):
    names = []

    for d in detections:
        clean = clean_class_name(d["class_name"])
        if is_not_leaf(clean):
            continue
        names.append(clean)

    unique_names = sorted(set(names))

    if not unique_names:
        log_text = "not_a_leaf"
    else:
        log_text = " ".join(unique_names)

    log_to_sheet(log_text)


def summarize_detections(detections):
    if not detections:
        return (
             "📷 ไม่สามารถวิเคราะห์ภาพได้อย่างแม่นยำ\n\n"
            "กรุณาส่งภาพใหม่ให้เห็นใบหรือบริเวณที่ผิดปกติชัดเจน "
            "และถ่ายในบริเวณที่มีแสงเพียงพอ"
        )

    lines = ["🌱 ผลการวิเคราะห์:\n"]

    for i, d in enumerate(detections, 1):
        class_name = clean_class_name(d["class_name"])
        display_name = display_class_name(class_name)
        detail = get_disease_detail(class_name)

        lines.append(
            f"{i}. {display_name}\n"
            f"📊 ความมั่นใจ: {d['confidence'] * 100:.2f}%\n"
            f"{detail}\n"
        )

    return "\n".join(lines)

#ฟังก์ชันวิเคราะห์ภาพ
def detect_and_classify(image_path, conf_yolo=YOLO_CONF):
    #อ่านภาพเข้ามา ใช้ OpenCV อ่านไฟล์ภาพจาก path ที่รับเข้ามา
    print("detect_and_classify: reading image")
    img_bgr = cv2.imread(image_path)

    #เช็กว่ารูปถูกอ่านสำเร็จหรือไม่
    if img_bgr is None:
        raise ValueError("ไม่สามารถอ่านไฟล์ภาพได้")

    #พิมพ์ขนาดภาพเดิมออกมาดู
    print("detect_and_classify: original shape =", img_bgr.shape)

    #ย่อภาพถ้าขนาดใหญ่เกินไป
    img_bgr = resize_if_needed(img_bgr, max_side=MAX_IMAGE_SIDE)
    print("detect_and_classify: resized shape =", img_bgr.shape)

    #เตรียมตัวแปรสำหรับใช้งานต่อ
    h, w = img_bgr.shape[:2] #เก็บความสูงและความกว้างของภาพไว้ใช้ภายหลัง
    result_img = img_bgr.copy() #สร้างสำเนาของภาพไว้สำหรับวาดกรอบและข้อความเพื่อไม่แก้ไขภาพต้นฉบับโดยตรง

    raw_detections = [] #สร้าง list ว่างไว้เก็บผล detection ที่ผ่านเงื่อนไขแล้ว

    print("detect_and_classify: running yolo")
    
    #ใช้ YOLO ตรวจหาบริเวณที่น่าสนใจ
    results = yolo_model(img_bgr, conf=conf_yolo, verbose=False)
    print("detect_and_classify: yolo done")

    boxes = results[0].boxes
    
    #ถ้าYOLOเจออย่างน้อย 1 กรอบ จึงค่อยเข้าส่วนนี้
    if boxes is not None and len(boxes) > 0:
        #นับจำนวนกล่องทั้งหมดที่ YOLO เจอ
        total_boxes = len(boxes)
        print("detect_and_classify: boxes found =", total_boxes)

        #จำกัดจำนวนกล่องที่จะประมวลผลจริงไม่เกิน MAX_BOXES
        num_to_process = min(total_boxes, MAX_BOXES)
        print("detect_and_classify: processing up to", num_to_process, "boxes")

        #วนทำงานทีละกรอบ
        for i, box in enumerate(boxes[:MAX_BOXES]):
            print("processing box", i + 1)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) #x1, y1 = มุมซ้ายบน x2, y2 = มุมขวาล่าง

            #เพิ่มพื้นที่รอบกรอบอีก 10 pixel ทุกด้าน เพื่อช่วยให้มองเห็นลักษณะโรคได้ดีขึ้น
            pad = 10
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            #ครอปภาพเฉพาะส่วนที่ตรวจพบ
            crop = img_bgr[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                print("skip empty crop")
                continue

            #ส่งภาพที่ครอปไปให้โมเดลจำแนกโรค
            print("classifying crop", i + 1, "shape =", crop.shape)
            idx, conf = classify_leaf(crop) #idx = index ของ class ที่ทำนาย ,conf = ความมั่นใจของโมเดล
            print("crop classified", i + 1, "conf =", conf)

            #แปลงเลข class เป็นชื่อโรค
            class_name = clean_class_name(class_names[idx])

            #ถ้าผลออกมาว่าไม่ใช่ใบ
            if is_not_leaf(class_name):
                print("skip not_a_leaf")
                continue

            #ค่าความมั่นใจต่ำเกินไป
            if conf < MIN_CONF:
                print("skip low conf:", conf)
                continue

            #เก็บผลที่ผ่านเงื่อนไขแล้ว
            raw_detections.append({
                "class_name": class_name,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })

    #ถ้า YOLO ไม่เจอส่วนที่เป็นใบเลย จะวิเคราะห์ทั้งภาพแทน
    if len(raw_detections) == 0:
        print("fallback classify whole image")
        idx, conf = classify_leaf(img_bgr)
        class_name = clean_class_name(class_names[idx])
        print("fallback class =", class_name, "conf =", conf)

        if not is_not_leaf(class_name) and conf >= MIN_CONF:
            raw_detections.append({
                "class_name": class_name,
                "confidence": conf,
                "box": (5, 5, w - 5, h - 5)
            })

    max_conf = max([d["confidence"] for d in raw_detections], default=0.0)
    print("max_conf =", max_conf)

    if max_conf < GLOBAL_CONF:
        print("below GLOBAL_CONF, return empty result")
        return [], None

    final_detections = raw_detections

    #วาดกรอบและข้อความลงบนภาพ
    for d in final_detections:
        x1, y1, x2, y2 = d["box"]
        class_name = d["class_name"]
        conf = d["confidence"]

        #เลือกสีกรอบ
        color = (0, 200, 0) if is_healthy(class_name) else (0, 0, 255)

        #วาดกรอบ
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

        #สร้างข้อความ
        label_text = f"{display_class_name(class_name)} {conf * 100:.1f}%"
        #วาดข้อความลงภาพ
        draw_label(result_img, label_text, x1, y1, color)

    #สร้างชื่อไฟล์ผลลัพธ์แบบสุ่ม
    result_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(STATIC_DIR, result_filename)
    cv2.imwrite(result_path, result_img)

    print("result image saved:", result_path)

    #คืนค่าผลลัพธ์กลับออกไป
    return final_detections, result_filename


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return "Doctor Tomato LINE Bot is running", 200


@app.route("/health")
def health():
    return "ok", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    # รองรับ verification request / body ว่าง
    if not body.strip():
        return "OK", 200

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    except Exception as e:
        print("❌ callback error:", str(e))
        traceback.print_exc()
        return "Error", 500

    return "OK", 200


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/favicon.ico")
def favicon():
    return "", 204


# =========================================================
# LINE HANDLER
# =========================================================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    input_path = None
    result_path = None

    try:
        print("=== handle_image: start ===")

        message_id = event.message.id
        print("message_id =", message_id)

        content = line_bot_api.get_message_content(message_id)
        print("got message content")

        input_filename = f"input_{message_id}.jpg"
        input_path = os.path.join(INPUT_DIR, input_filename)

        with open(input_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        print("saved input image:", input_path)

        detections, result_filename = detect_and_classify(input_path)
        print("detect_and_classify finished")

        if not detections:
            reply_text = summarize_detections(detections)
            print("no detections, replying text only")
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            print("reply sent")
            return

        log_multiple_diseases(detections)
        print("logged to sheet")

        reply_text = summarize_detections(detections)

        if not PUBLIC_BASE_URL:
            raise ValueError("PUBLIC_BASE_URL is not set")

        image_url = f"{PUBLIC_BASE_URL}/static/{result_filename}"
        result_path = os.path.join(STATIC_DIR, result_filename)

        print("image_url =", image_url)

        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=reply_text),
                ImageSendMessage(
                    original_content_url=image_url,
                    preview_image_url=image_url
                )
            ]
        )
        print("reply with image sent")

    except Exception as e:
        print("❌ handle_image error:", str(e))
        traceback.print_exc()
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ\n{str(e)}"
                )
            )
        except Exception as inner_e:
            print("❌ reply error:", str(inner_e))

    finally:
        # ลบ input file
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
                print("deleted input file:", input_path)
        except Exception as e:
            print("⚠️ failed to delete input file:", e)

        # ยังไม่ลบ result file ทันที เพราะ LINE อาจยังต้องโหลดรูปจาก URL
        # ถ้าจะลบภายหลัง ค่อยทำเป็น scheduled cleanup ภายหลัง


# =========================================================
# STARTUP
# =========================================================
init_google_sheet()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
