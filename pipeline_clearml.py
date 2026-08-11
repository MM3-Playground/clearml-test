from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
from typing import Any
from clearml import Dataset, Task
from clearml.automation.controller import PipelineDecorator

DEFAULT_QUEUE=os.getenv('CLEARML_PIPELINE_QUEUE','default')
DEFAULT_DOCKER=os.getenv('CLEARML_TASK_DOCKER','python:3.11')

def _read_manifest(path):
    rows=[]
    for n,raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not raw.strip(): continue
        p=raw.rstrip().split('\t')
        if len(p)!=2: raise ValueError(f'Invalid manifest line {n}: expected PATH<TAB>LABEL')
        rows.append((p[0],int(p[1])))
    return rows

def _portable_manifest(source,destination,dataset_root=''):
    root=Path(dataset_root).expanduser().resolve() if dataset_root else None
    dest=Path(destination); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8') as f:
        for raw_path,label in _read_manifest(source):
            p=Path(raw_path).expanduser(); rendered=p.as_posix()
            if root is not None and p.is_absolute(): rendered=p.resolve().relative_to(root).as_posix()
            f.write(f'{rendered}\t{label}\n')
    return dest

def upload_manifest_bundle(config):
    task=Task.init(project_name=config['clearml_project_name'],task_name=config.get('task_name','manifest-bundle'),task_type=Task.TaskTypes.data_processing,output_uri=True)
    root=str(config.get('dataset_root','') or '')
    with tempfile.TemporaryDirectory(prefix='clearml-manifests-') as tmp:
        for key,name in [('train_paths_file','train.txt'),('val_paths_file','val.txt'),('test_paths_file','test.txt')]:
            p=_portable_manifest(config[key],Path(tmp)/name,root)
            task.upload_artifact(f'manifest_{key}',artifact_object=str(p),wait_on_upload=True)
    task.close(); return task.id

def _resolve_dataset_root(dataset_id,persistent_dataset_path):
    if persistent_dataset_path:
        p=Path(persistent_dataset_path).expanduser().resolve()
        if not p.is_dir(): raise FileNotFoundError(f'Persistent dataset is not available: {p}')
        return p,'persistent'
    if not dataset_id: raise ValueError('Either dataset_id or persistent_dataset_path is required')
    ds=Dataset.get(dataset_id=dataset_id,alias='dataset',overridable=True)
    return Path(ds.get_local_copy()).resolve(),'clearml'

def _materialize_manifests(manifest_task_id,root,dest):
    t=Task.get_task(task_id=manifest_task_id); dest.mkdir(parents=True,exist_ok=True)
    out={}
    for key,file in [('train','train.txt'),('val','val.txt'),('test','test.txt')]:
        src=Path(t.artifacts[f'manifest_{key}_paths_file'].get_local_copy()); target=dest/file
        with target.open('w',encoding='utf-8') as f:
            for raw in src.read_text(encoding='utf-8').splitlines():
                if not raw.strip(): continue
                rel,label=raw.rstrip().split('\t',1); p=Path(rel).expanduser(); actual=p if p.is_absolute() else root/p
                f.write(f'{actual.resolve()}\t{label}\n')
        out[key]=str(target.resolve())
    return out

@PipelineDecorator.component(name='train',return_values=['training'],cache=False,execution_queue=DEFAULT_QUEUE,docker=DEFAULT_DOCKER,repo='.',packages=False)
def train_component(dataset_id:str,persistent_dataset_path:str,manifest_task_id:str,clearml_project_name:str,clearml_task_name:str,run_name:str,model:str,image_size:int,batch_size:int,workers:int,n_epochs:int,lr:float,factor:float,patience:int):
    import json,subprocess,sys
    from pathlib import Path
    from clearml import Task
    task=Task.current_task(); root,mode=_resolve_dataset_root(dataset_id,persistent_dataset_path)
    task.set_parameter('dataset/mode',mode); task.set_parameter('dataset/id',dataset_id); task.set_parameter('dataset/persistent_path',persistent_dataset_path)
    work=Path(os.getenv('TMPDIR','/tmp'))/'clearml-pipeline'/task.id; manifests=_materialize_manifests(manifest_task_id,root,work/'manifests'); save=work/'runs'
    cmd=[sys.executable,'-u','train_torch_test.py','--run_name',run_name,'--save_dir',str(save),'--paths_file',manifests['train'],'--val_paths_file',manifests['val'],'--image_size',str(image_size),'--batch_size',str(batch_size),'--workers',str(workers),'--model',model,'--lr',str(lr),'--n_epochs',str(n_epochs),'--factor',str(factor),'--patience',str(patience),'--clearml_project_name',clearml_project_name,'--clearml_task_name',clearml_task_name]
    if dataset_id: cmd += ['--clearml_dataset_id',dataset_id]
    subprocess.run(cmd,check=True); result=json.loads((save/'training-result.json').read_text()); result['dataset_mode']=mode; return result

@PipelineDecorator.component(name='evaluate',return_values=['evaluation'],cache=False,execution_queue=DEFAULT_QUEUE,docker=DEFAULT_DOCKER,repo='.',packages=False)
def evaluate_component(dataset_id:str,persistent_dataset_path:str,manifest_task_id:str,training:dict,clearml_project_name:str,clearml_task_name:str,model:str,image_size:int,minimum_accuracy:float):
    import json,subprocess,sys
    from pathlib import Path
    from clearml import InputModel,Task
    task=Task.current_task(); root,mode=_resolve_dataset_root(dataset_id,persistent_dataset_path); work=Path(os.getenv('TMPDIR','/tmp'))/'clearml-pipeline'/task.id
    manifests=_materialize_manifests(manifest_task_id,root,work/'manifests'); model_path=InputModel(model_id=training['model_id']).get_local_copy(); out=work/'evaluation'
    subprocess.run([sys.executable,'-u','eval.py','--iut_paths_file',manifests['test'],'--image_size',str(image_size),'--out_dir',str(out),'--model',model,'--load_path',str(model_path),'--device','cpu','--clearml_project_name',clearml_project_name,'--clearml_task_name',f'{clearml_task_name}-evaluation','--parent_training_task_id',str(training['training_task_id'])],check=True)
    r=json.loads((out/'result.json').read_text()); r['minimum_accuracy']=float(minimum_accuracy); r['accepted']=float(r['accuracy'])>=float(minimum_accuracy); r['dataset_mode']=mode; task.get_logger().report_single_value('accepted',int(r['accepted'])); return r

@PipelineDecorator.pipeline(name='clearml-training-pipeline',project='ClearML Pipelines',version='4.0.0',default_queue=DEFAULT_QUEUE,pipeline_execution_queue=DEFAULT_QUEUE,abort_on_failure=True)
def training_pipeline(dataset_id:str='',persistent_dataset_path:str='',manifest_task_id:str='',clearml_project_name:str='clearml-orchestration-demo',clearml_task_name:str='cpu-demo',run_name:str='cpu-demo',model:str='ours',image_size:int=128,batch_size:int=1,workers:int=0,n_epochs:int=2,lr:float=.001,factor:float=.9,patience:int=5,minimum_accuracy:float=0.0):
    tr=train_component(dataset_id,persistent_dataset_path,manifest_task_id,clearml_project_name,clearml_task_name,run_name,model,image_size,batch_size,workers,n_epochs,lr,factor,patience)
    return evaluate_component(dataset_id,persistent_dataset_path,manifest_task_id,tr,clearml_project_name,clearml_task_name,model,image_size,minimum_accuracy)

def run_config(cfg): return training_pipeline(**cfg)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    m=sub.add_parser('manifests'); m.add_argument('--config',required=True)
    r=sub.add_parser('run'); r.add_argument('--config',required=True); r.add_argument('--mode',choices=['local','remote'],default='remote'); r.add_argument('--queue',default=DEFAULT_QUEUE)
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    if a.cmd=='manifests': print(json.dumps({'manifest_task_id':upload_manifest_bundle(cfg)},indent=2)); return
    if a.mode=='local':
        # Same decorated DAG, but components execute on this machine/allocation rather than an Agent.
        PipelineDecorator.run_locally()
    else:
        # Initial bootstrap: controller is handed to the worker queue; laptop does not train.
        task=Task.init(project_name='ClearML Pipelines',task_name='clearml-training-pipeline-bootstrap',task_type=Task.TaskTypes.controller)
        task.execute_remotely(queue_name=a.queue,clone=False,exit_process=False)
    run_config(cfg)

if __name__=='__main__': main()
