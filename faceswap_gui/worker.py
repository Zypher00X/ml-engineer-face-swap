"""JSON-lines protocol; ordinary model logging goes to stderr."""
import sys,json,time,contextlib,traceback
from pathlib import Path

def main():
    source,target,output=sys.argv[1:4]
    timeline=[];started=time.monotonic()
    def emit(stage,state):
        event={'type':'stage','stage':stage,'state':state,'elapsed':time.monotonic()-started}
        timeline.append(event);print(json.dumps(event),flush=True)
    try:
        emit('load','started')
        with contextlib.redirect_stdout(sys.stderr):
            from faceswap.pipeline import Config,create_pipeline
            pipeline=create_pipeline(Config(backend='hyperswap',input_sampling='area',save_debug=True))
        emit('load','completed')
        # Keep progress JSON on the original stdout while model libraries log to stderr.
        protocol=sys.stdout
        def progress(stage,state):
            with contextlib.redirect_stdout(protocol):emit(stage,state)
        with contextlib.redirect_stdout(sys.stderr):
            _,_,meta=pipeline.swap(source,target,output,progress=progress)
        Path(output).with_suffix('.events.json').write_text(json.dumps(timeline,indent=2),encoding='utf-8')
        print(json.dumps({'type':'result','output':output,'seconds':time.monotonic()-started,
                          'gate_pass':meta['quality']['project_gate_pass']}),flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'type':'error','message':str(exc)}),flush=True)
        return 1

if __name__=='__main__':sys.exit(main())
