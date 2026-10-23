import sys,json,time,uuid,shutil,os
from pathlib import Path
from PySide6.QtCore import Qt,QProcess,QProcessEnvironment,QTimer,QThread,Signal
from PySide6.QtGui import QPixmap,QFont,QImageReader,QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
    QLabel,QPushButton,QFrame,QFileDialog,QMessageBox,QProgressBar,QDialog,QScrollArea,QSlider)

ROOT=Path(__file__).resolve().parents[1]
STAGES={'load':'โหลดโมเดลและเตรียมอุปกรณ์','read':'อ่านไฟล์ภาพ',
        'source':'ตรวจใบหน้าต้นฉบับ','target':'ตรวจใบหน้าเป้าหมายและจัดแนว',
        'generate':'สร้างใบหน้าด้วย HyperSwap 256',
        'blend':'แยกส่วนใบหน้าและผสานภาพ','validate':'ตรวจวัดผลอัตโนมัติ','save':'บันทึกภาพและข้อมูลตรวจสอบ'}
STYLE='''
QWidget { background:#101419; color:#eef2f7; font-family:"Leelawadee UI"; font-size:14px; }
QFrame#card { background:#191f27; border:1px solid #303b49; border-radius:12px; }
QFrame#card QLabel { background:transparent; border:none; }
QLabel#title { font-size:25px; font-weight:700; }
QLabel#heading { font-size:21px; font-weight:700; }
QLabel#muted { color:#aab8ca; }
QLabel#preview { background:#11171e; border:1px dashed #3c4c60; border-radius:8px; color:#8fa4bd; }
QPushButton { background:#222c38; border:1px solid #41536a; border-radius:8px; padding:10px 18px; font-weight:600; }
QPushButton:hover { background:#2e3d4e; border-color:#36bdc5; }
QPushButton:disabled { color:#69798c; border-color:#2b3542; background:#1b242f; }
QPushButton#primary { background:#079caa; border-color:#16b9c5; color:white; font-size:20px; padding:13px; }
QPushButton#primary:hover { background:#0aafb9; }
QPushButton#primary:disabled { background:#24565c; color:#9caeb8; }
QProgressBar { background:#253447; border:none; border-radius:5px; height:12px; text-align:center; }
QProgressBar::chunk { background:#12b7bf; border-radius:5px; }
QProgressBar#cpu::chunk { background:#35a4ff; }
QProgressBar#gpu::chunk { background:#9761ff; }
QScrollArea { border:none; }
'''

class Preview(QLabel):
    clicked=Signal()
    def __init__(self,text):
        super().__init__(text);self.setObjectName('preview');self.setAlignment(Qt.AlignCenter)
        self.original=QPixmap();self.setMinimumSize(220,155)
    def set_image(self,path):
        reader=QImageReader(str(path));reader.setAutoTransform(True)
        image=reader.read()
        if image.isNull():raise ValueError('ไม่สามารถเปิดภาพนี้ได้: '+reader.errorString())
        self.original=QPixmap.fromImage(image);self.refresh()
    def clear_image(self,text):self.original=QPixmap();self.setPixmap(QPixmap());self.setText(text)
    def refresh(self):
        if not self.original.isNull():self.setPixmap(self.original.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if not self.original.isNull():self.setPixmap(self.original.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:self.clicked.emit()

class ImageCard(QFrame):
    changed=Signal()
    def __init__(self,title,subtitle):
        super().__init__();self.setObjectName('card');self.path=None
        layout=QVBoxLayout(self);layout.setContentsMargins(18,14,18,14)
        heading=QLabel(title);heading.setObjectName('heading');layout.addWidget(heading)
        sub=QLabel(subtitle);sub.setObjectName('muted');layout.addWidget(sub)
        row=QHBoxLayout();self.preview=Preview('คลิกเพื่อเลือกรูปภาพ');row.addWidget(self.preview,3)
        controls=QVBoxLayout();self.file=QLabel('ยังไม่ได้เลือกไฟล์');self.file.setWordWrap(True)
        self.choose=QPushButton('เลือกรูปภาพ');self.choose.clicked.connect(self.pick)
        self.preview.clicked.connect(self.pick);controls.addStretch();controls.addWidget(self.file);controls.addWidget(self.choose);controls.addStretch()
        row.addLayout(controls,2);layout.addLayout(row,1)
    def pick(self):
        if not self.choose.isEnabled():return
        path,_=QFileDialog.getOpenFileName(self,'เลือกรูปภาพ',str(ROOT),'รูปภาพ (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            try:self.select(path)
            except ValueError as exc:QMessageBox.warning(self,'เปิดภาพไม่ได้',str(exc))
    def select(self,path):
        path=Path(path).resolve();self.preview.set_image(path);self.path=path
        size=self.preview.original.size();self.file.setText(f'{path.name}\n{size.width()} × {size.height()} px\n{path.stat().st_size/1048576:.2f} MB')
        self.file.setToolTip(str(path));self.changed.emit()

class TelemetryThread(QThread):
    sample_ready=Signal(dict)
    def run(self):
        from .telemetry import WindowsCounters
        counter=None
        try:
            counter=WindowsCounters()
            while not self.isInterruptionRequested():
                self.msleep(1000)
                if not self.isInterruptionRequested():self.sample_ready.emit(counter.sample())
        except Exception as exc:self.sample_ready.emit({'cpu':None,'gpu':None,'error':str(exc)})
        finally:
            if counter:counter.close()

class FullImage(QDialog):
    def __init__(self,path,parent=None):
        super().__init__(parent);self.setWindowTitle('ภาพผลลัพธ์ — '+Path(path).name);self.resize(1050,800)
        self.pix=QPixmap(str(path));layout=QVBoxLayout(self);scroll=QScrollArea();self.label=QLabel();scroll.setWidget(self.label)
        layout.addWidget(scroll);row=QHBoxLayout();row.addWidget(QLabel('ซูม'))
        slider=QSlider(Qt.Horizontal);slider.setRange(10,300);slider.setValue(100);slider.valueChanged.connect(self.zoom)
        row.addWidget(slider);self.scale_text=QLabel();row.addWidget(self.scale_text);layout.addLayout(row);self.zoom(100)
    def zoom(self,percent):
        self.label.setPixmap(self.pix.scaled(self.pix.size()*percent/100,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.label.adjustSize();self.scale_text.setText(f'{percent}%')

class MainWindow(QMainWindow):
    def __init__(self,telemetry=True):
        super().__init__();self.setWindowTitle('ML engineer Project face swap');self.resize(1440,940)
        self.setMinimumSize(900,640);self.running=False;self.result_path=None;self.events=[];self.telemetry_samples=[];self.completed=set()
        self.process=None;self.stdout_buffer='';self.pending_result=None;self.error_message=None;self.dialogs=[]
        outer=QWidget();outer.setMinimumSize(1040,850)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(outer);self.setCentralWidget(scroll)
        layout=QVBoxLayout(outer);layout.setContentsMargins(22,16,22,16);layout.setSpacing(14)
        header=QHBoxLayout();title=QLabel('ML engineer Project face swap');title.setObjectName('title');header.addWidget(title);header.addStretch()
        badge=QLabel('HyperSwap 256  •  AREA');badge.setStyleSheet('color:#47d8dd; padding:9px; background:#153136; border-radius:10px;');header.addWidget(badge);layout.addLayout(header)
        inputs=QHBoxLayout();self.source=ImageCard('01   ใบหน้าต้นฉบับ','ใช้ตัวตนจากภาพนี้');self.target=ImageCard('02   ภาพเป้าหมาย','ใช้ท่าศีรษะและสีหน้าจากภาพนี้')
        for card in (self.source,self.target):inputs.addWidget(card,1);card.changed.connect(self.input_changed)
        layout.addLayout(inputs,5)
        self.start_button=QPushButton('เริ่มสลับใบหน้า');self.start_button.setObjectName('primary');self.start_button.setEnabled(False);self.start_button.clicked.connect(self.start_job);layout.addWidget(self.start_button)
        bottom=QHBoxLayout();status=QFrame();status.setObjectName('card');left=QVBoxLayout(status);left.setContentsMargins(18,14,18,14)
        self.status_label=QLabel('พร้อมเริ่มงาน');self.status_label.setObjectName('heading');left.addWidget(self.status_label)
        self.stage_labels={}
        for stage,text in STAGES.items():
            label=QLabel('[ ]  '+text);label.setStyleSheet('color:#9cacc0;');self.stage_labels[stage]=label;left.addWidget(label)
        self.progress=QProgressBar();self.progress.setRange(0,len(STAGES));self.progress.setValue(0);self.progress.setTextVisible(False);left.addWidget(self.progress)
        self.detail=QLabel('เสร็จแล้ว 0/8 ขั้นตอน • ยังไม่เริ่มประมวลผล');self.detail.setWordWrap(True);left.addWidget(self.detail)
        self.elapsed=QLabel('เวลา 0.0 วินาที');self.elapsed.setObjectName('muted');left.addWidget(self.elapsed)
        scope=QLabel('ทรัพยากรรวมทั้งเครื่อง • อ่าน Windows ทุก 1 วินาที');scope.setObjectName('muted');left.addWidget(scope)
        meters=QHBoxLayout();self.meters={};self.meter_labels={}
        for name in ('CPU','GPU'):
            column=QVBoxLayout();label=QLabel(name+'  —');bar=QProgressBar();bar.setObjectName(name.lower());bar.setRange(0,100);bar.setValue(0);bar.setTextVisible(False)
            column.addWidget(label);column.addWidget(bar);meters.addLayout(column);self.meters[name]=bar;self.meter_labels[name]=label
        left.addLayout(meters);self.memory=QLabel('หน่วยความจำ GPU: รอข้อมูล');self.memory.setWordWrap(True);self.memory.setObjectName('muted');left.addWidget(self.memory)
        self.counter_note=QLabel('CPU: Processor Utility • GPU: engine ที่ใช้งานสูงสุด');self.counter_note.setWordWrap(True);self.counter_note.setObjectName('muted');left.addWidget(self.counter_note)
        bottom.addWidget(status,6)
        result=QFrame();result.setObjectName('card');right=QVBoxLayout(result);right.setContentsMargins(18,14,18,14)
        heading=QLabel('ผลลัพธ์');heading.setObjectName('heading');right.addWidget(heading)
        self.result_preview=Preview('ผลลัพธ์จะแสดงที่นี่');self.result_preview.clicked.connect(self.open_full);right.addWidget(self.result_preview,1)
        buttons=QHBoxLayout();self.full_button=QPushButton('ดูภาพเต็ม');self.save_button=QPushButton('บันทึกภาพ')
        self.full_button.clicked.connect(self.open_full);self.save_button.clicked.connect(self.save_image)
        for button in (self.full_button,self.save_button):button.setEnabled(False);buttons.addWidget(button)
        right.addLayout(buttons);bottom.addWidget(result,5);layout.addLayout(bottom,6)
        footer=QLabel('โมเดล ONNX ใช้ CPU • แยกส่วนภาพใช้ CUDA เมื่อพร้อม • ผลวัดอัตโนมัติแยกจากการตรวจภาพด้วยตา');footer.setObjectName('muted');layout.addWidget(footer)
        self.clock=QTimer(self);self.clock.setInterval(100);self.clock.timeout.connect(self.tick)
        self.monitor=None
        if telemetry:
            self.monitor=TelemetryThread(self);self.monitor.sample_ready.connect(self.update_telemetry);self.monitor.start()

    def input_changed(self):
        if self.running:return
        self.result_path=None;self.result_preview.clear_image('ผลลัพธ์จะแสดงที่นี่')
        self.full_button.setEnabled(False);self.save_button.setEnabled(False)
        self.start_button.setEnabled(bool(self.source.path and self.target.path))
    def start_job(self):
        if self.running or not self.source.path or not self.target.path:return
        self.running=True;self.result_path=None;self.events=[];self.completed=set();self.pending_result=None;self.error_message=None;self.stdout_buffer=''
        self.progress.setValue(0);self.result_preview.clear_image('กำลังประมวลผล…');self.status_label.setText('กำลังเริ่มงาน')
        for stage,label in self.stage_labels.items():label.setText('[ ]  '+STAGES[stage]);label.setStyleSheet('color:#9cacc0;')
        for button in (self.start_button,self.source.choose,self.target.choose,self.full_button,self.save_button):button.setEnabled(False)
        self.job_folder=ROOT/'outputs/gui'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);self.job_folder.mkdir(parents=True)
        self.log=open(self.job_folder/'worker.log','wb');self.started=time.monotonic();self.started_wall=time.time();self.clock.start()
        self.process=QProcess(self);self.process.setWorkingDirectory(str(ROOT))
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONIOENCODING','utf-8');env.insert('PYTHONUNBUFFERED','1');self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_stdout);self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.job_finished);self.process.errorOccurred.connect(self.process_error)
        python=ROOT/'venv/Scripts/python.exe'
        self.process.start(str(python),['-X','utf8','-m','faceswap_gui.worker',str(self.source.path),str(self.target.path),str(self.job_folder/'result.png')])
    def tick(self):self.elapsed.setText(f'เวลา {time.monotonic()-self.started:.1f} วินาที')
    def read_stderr(self):
        if self.log and not self.log.closed:self.log.write(bytes(self.process.readAllStandardError()));self.log.flush()
    def read_stdout(self):
        self.stdout_buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.stdout_buffer:
            line,self.stdout_buffer=self.stdout_buffer.split('\n',1)
            try:self.handle_event(json.loads(line))
            except (ValueError,KeyError):pass
    def handle_event(self,event):
        self.events.append(event)
        if event['type']=='stage':
            stage=event['stage'];done=event['state']=='completed'
            self.stage_labels[stage].setText(('[OK]  ' if done else '[...]  ')+STAGES[stage])
            self.stage_labels[stage].setStyleSheet('color:#5cddaa;' if done else 'color:#4cd6df;')
            if done:self.completed.add(stage)
            self.progress.setValue(len(self.completed));self.detail.setText(f'เสร็จแล้ว {len(self.completed)}/8 ขั้นตอน • ความคืบหน้าตามขั้นตอนจริง')
            self.status_label.setText(STAGES[stage])
        elif event['type']=='result':self.pending_result=event
        elif event['type']=='error':self.error_message=event['message']
    def process_error(self,error):
        if error==QProcess.FailedToStart:
            self.error_message='เปิด worker ไม่ได้ ตรวจ venv และไฟล์โปรแกรม';self.job_finished(-1,QProcess.CrashExit)
    def job_finished(self,code,status):
        if not self.running:return
        self.read_stdout();self.read_stderr();self.log.close();self.clock.stop();self.tick();self.running=False
        self.source.choose.setEnabled(True);self.target.choose.setEnabled(True);self.start_button.setEnabled(True)
        if code==0 and self.pending_result:
            self.result_path=Path(self.pending_result['output']);self.result_preview.set_image(self.result_path)
            self.result_preview.setToolTip(str(self.result_path))
            self.full_button.setEnabled(True);self.save_button.setEnabled(True);self.status_label.setText('เสร็จสมบูรณ์')
            self.detail.setText('เสร็จแล้ว 8/8 ขั้นตอน • '+('คะแนนอัตโนมัติผ่าน — โปรดตรวจภาพด้วยตา' if self.pending_result['gate_pass'] else 'มีคะแนนไม่ผ่านเกณฑ์ — โปรดตรวจ JSON และภาพ'))
        else:
            self.status_label.setText('ประมวลผลไม่สำเร็จ');self.result_preview.clear_image('ไม่มีผลลัพธ์จากงานนี้')
            message=self.error_message or f'Worker หยุดทำงาน (exit {code})'
            if 'detected 0' in message:message='ตรวจไม่พบใบหน้า กรุณาใช้ภาพที่เห็นใบหน้าชัดเจน'
            self.detail.setText(message);self.detail.setToolTip(str(self.job_folder/'worker.log'))
        (self.job_folder/'gui_events.json').write_text(json.dumps(self.events,ensure_ascii=False,indent=2),encoding='utf-8')
        job_samples=[sample for sample in self.telemetry_samples if sample.get('timestamp',0)>=self.started_wall]
        (self.job_folder/'telemetry.json').write_text(json.dumps(job_samples,ensure_ascii=False,indent=2),encoding='utf-8')
        self.process.deleteLater()
    def update_telemetry(self,sample):
        self.telemetry_samples.append(sample);self.telemetry_samples=self.telemetry_samples[-3600:]
        for name in ('CPU','GPU'):
            value=sample.get(name.lower());self.meter_labels[name].setText(name+('  ไม่พร้อมใช้งาน' if value is None else f'  {value:.1f}%'))
            self.meters[name].setValue(0 if value is None else round(value));self.meters[name].setEnabled(value is not None)
        memories=sample.get('dedicated_bytes',{})
        self.memory.setText('GPU dedicated: '+(' • '.join(f'อะแดปเตอร์ {i+1}: {v/1073741824:.2f} GB' for i,(k,v) in enumerate(sorted(memories.items())) if v>0) if any(v>0 for v in memories.values()) else 'ไม่พร้อมใช้งาน / ไม่ได้ใช้หน่วยความจำ dedicated'))
        self.memory.setToolTip(json.dumps(memories));self.counter_note.setToolTip(json.dumps(sample,ensure_ascii=False))
    def open_full(self):
        if self.result_path:
            dialog=FullImage(self.result_path,self);self.dialogs.append(dialog);dialog.show()
    def export_image(self,path):
        if not self.result_path:raise ValueError('ยังไม่มีภาพผลลัพธ์')
        path=Path(path).resolve()
        if path in (self.source.path,self.target.path):raise ValueError('ห้ามบันทึกทับภาพต้นฉบับหรือเป้าหมาย')
        if path.suffix.lower()!='.png':raise ValueError('กรุณาบันทึกเป็น PNG')
        if path!=self.result_path.resolve():shutil.copyfile(self.result_path,path)
    def save_image(self):
        if not self.result_path:return
        path,_=QFileDialog.getSaveFileName(self,'บันทึกภาพ',str(ROOT/'outputs/face_swap.png'),'PNG (*.png)')
        if path:
            if not Path(path).suffix:path+='.png'
            try:self.export_image(path)
            except (ValueError,OSError) as exc:QMessageBox.warning(self,'บันทึกไม่ได้',str(exc))
    def closeEvent(self,event):
        if self.running:
            QMessageBox.information(self,'กำลังประมวลผล','กรุณารอให้งานเสร็จก่อนปิดโปรแกรม');event.ignore();return
        if self.monitor:self.monitor.requestInterruption();self.monitor.wait(2500)
        for dialog in self.dialogs:dialog.close()
        event.accept()

def configure_app(app):
    # Explicit font loading also makes offscreen QA render real Thai glyphs.
    folder=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for name in ('LeelawUI.ttf','LeelaUIb.ttf'):
        path=folder/name
        if path.is_file():QFontDatabase.addApplicationFont(str(path))
    app.setStyle('Fusion');app.setStyleSheet(STYLE);app.setFont(QFont('Leelawadee UI',10))

def main():
    app=QApplication(sys.argv);configure_app(app)
    window=MainWindow();window.show();sys.exit(app.exec())
