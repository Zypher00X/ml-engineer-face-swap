# ML engineer Project face swap

GUI Windows โทนดำสำหรับ HyperSwap 256 ใช้ pipeline เดียวกับ CLI พร้อม `input_sampling=area`, `protect_bridge=False`, `save_debug=True` ไม่มีการเปลี่ยนอัลกอริทึมสลับใบหน้าใน GUI

## เปิดโปรแกรม

ดับเบิลคลิก `เปิดโปรแกรม Face Swap.bat` ที่โฟลเดอร์โปรเจค หรือเปิด PowerShell ใน `C:\ML_2_3\LivePortrait` แล้วรัน:

```powershell
.\venv\Scripts\python.exe run_faceswap_gui.py
```

1. เลือกภาพ A ในช่องใบหน้าต้นฉบับ และ B ในช่องภาพเป้าหมาย
2. กดเริ่มสลับใบหน้า ระหว่างทำงานยังเลื่อน/ขยับหน้าต่างได้ แต่จะล็อกการเปลี่ยนภาพจนงานเสร็จ
3. คลิกภาพผลลัพธ์หรือดูภาพเต็ม เพื่อเปิดภาพ 100% พร้อมแถบซูม
4. กดบันทึกภาพ เลือกที่เก็บ PNG โปรแกรมป้องกันบันทึกทับ A/B

ผลต้นฉบับของทุกงานอยู่ที่ `outputs/gui/<วันเวลา>_<รหัสงาน>/` มี result.png, result.json, result_mask.png, result_debug/, result.events.json, gui_events.json, worker.log และ telemetry.json การส่งออกภาพไม่เปลี่ยนไฟล์ต้นฉบับของงาน กรณีล้มเหลวเก็บ log ไว้และปลดปุ่มให้ลองใหม่

หากมีหลายใบหน้า ใช้ใบหน้าที่มีพื้นที่มากที่สุดเช่นเดียวกับ CLI index=0 ยังไม่มีตัวเลือกใบหน้ารายบุคคลใน GUI รุ่นนี้ โปรแกรมใช้ venv และโมเดลในโปรเจค จึงยังไม่ใช่ EXE แบบย้ายไปเครื่องอื่นได้ทันที

## สถานะและทรัพยากร

- Worker แยก process เพื่อไม่ให้หน้าจอค้าง สถานะ started/completed มาจากขอบเขตงานจริง: โหลดโมเดล → อ่านภาพ → เตรียม A → เตรียม B → inference → segmentation/blend → diagnostics → บันทึก รวม 8 ขั้นตอน
- ความคืบหน้านับจำนวนขั้นตอนที่สำเร็จ ไม่ใช่เปอร์เซ็นต์เวลาที่เหลือ หากโมเดลล้มเหลวไม่ทำเครื่องหมายขั้นนั้นว่าสำเร็จ
- CPU/GPU เป็น **การใช้รวมทั้งเครื่อง** รวมโปรแกรมอื่น ไม่ใช่เปอร์เซ็นต์เฉพาะ face swap อ่าน Windows PDH ทุกประมาณ 1 วินาที ไม่ใช้ตัวเลขจำลอง
- CPU ใช้ `Processor Information(_Total)\% Processor Utility`; GPU ใช้ `GPU Engine(*)\Utilization Percentage` รวม process ที่ใช้ engine เดียวกัน แล้วเลือก engine ที่สูงที่สุดจากทุกอะแดปเตอร์ ไม่รวมทุก engine เป็นเปอร์เซ็นต์เดียว
- หน่วยความจำ dedicated อ่านจาก `GPU Adapter Memory(*)\Dedicated Usage` แยกอะแดปเตอร์ ไม่ใช่ torch.cuda.memory_allocated และไม่อ้างว่าเป็นความจุทั้งหมด
- หากอ่านไม่ได้จะแสดง “ไม่พร้อมใช้งาน” และปิดแถบ แทนการแสดงศูนย์เสมือนว่าไม่มีการใช้งาน
- ONNX generator และ InsightFace ใช้ CPU ตาม pipeline เดิม ส่วน parser ใช้ CUDA เมื่อมี จึงไม่คาดหวัง GPU 100% ตลอดงาน

ตัวเลขอาจไม่ตรง Task Manager ทุกจังหวะเพราะช่วงการสุ่ม/ความถี่อัปเดตต่างกัน บนหลาย GPU ให้เทียบ engine/อะแดปเตอร์ที่ใช้สูงสุด อย่าเทียบกับกราฟ 3D เพียงกราฟเดียว [Microsoft อธิบายการนับ GPU ตาม busiest engine](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)

## ติดตั้งซ้ำและทดสอบ

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_gui.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -X utf8 experiments/gui_validation.py
.\venv\Scripts\python.exe -X utf8 experiments/gui_ui_checks.py
.\venv\Scripts\python.exe -X utf8 experiments/gui_telemetry_check.py
```

`gui_validation.py` ใช้ Qt offscreen เป็นค่าเริ่มต้น หากต้องทดสอบด้วยหน้าต่าง Windows จริง ให้ตั้ง `$env:QT_QPA_PLATFORM='windows'` ก่อนรัน ตัวเลือกไฟล์ใช้ dialog ของ Qt; ชุดทดสอบแทนผลลัพธ์ dialog เพื่อคลิกทดสอบได้อัตโนมัติ

รายงานและภาพ QA อยู่ที่ `experiments/gui_validation/REPORT.md` การผ่าน integration ไม่ใช่การรับรองคุณภาพใบหน้า: JSON ยังคงแยก visual_acceptance เป็น pending_human_review
