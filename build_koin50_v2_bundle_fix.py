import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('downloaded_v2');out=Path('koin50_v2_bundle');out.mkdir(exist_ok=True)
router={};summaries=[];mont=[]
for sh in range(25):
    art=root/f'koin50-v2-shard-{sh}'
    hits=list(art.rglob(f'summary_{sh:02d}.json'))
    if not hits:
        hits=list(root.rglob(f'summary_{sh:02d}.json'))
    if not hits: raise SystemExit(f'missing shard {sh} summary')
    d=hits[0].parent
    s=json.load(open(hits[0]));summaries.append(s)
    pts=list(d.glob(f'koin50_v2_shard_{sh:02d}.pt'))
    if not pts: raise SystemExit(f'missing shard {sh} model')
    pt=pts[0];shutil.copy2(pt,out/pt.name);shutil.copy2(hits[0],out/hits[0].name)
    for local,cl in enumerate(s['global_class_ids']): router[cl]={'shard':sh,'local_class':local,'model':pt.name}
    p=d/f'samples_{sh:02d}.png'
    if p.exists():mont.append((sh,Image.open(p).convert('RGB')))
agg={'identity_classes':len(router),'total_accepted_face_crops':sum(x['accepted_images'] for x in summaries),'total_train_images':sum(x['train_images'] for x in summaries),'total_validation_images':sum(x['val_images'] for x in summaries),'mean_val_identity_accuracy':sum(x['val_identity_accuracy'] for x in summaries)/len(summaries),'min_val_identity_accuracy':min(x['val_identity_accuracy'] for x in summaries),'max_val_identity_accuracy':max(x['val_identity_accuracy'] for x in summaries),'mean_val_reconstruction_l1':sum(x['val_reconstruction_l1'] for x in summaries)/len(summaries),'total_model_parameters_across_shards':sum(x['parameters'] for x in summaries),'resolution':128,'shards':25,'per_shard':summaries,'raw_source_photos_uploaded':False,'usage_scope':'KoIn official README: academic purposes'}
(out/'router.json').write_text(json.dumps(router,indent=2));(out/'aggregate_summary.json').write_text(json.dumps(agg,indent=2))
if mont:
    w=max(im.width for _,im in mont);h=sum(im.height+20 for _,im in mont);can=Image.new('RGB',(w,h),'white');dr=ImageDraw.Draw(can);y=0
    for sh,im in mont:
        dr.text((2,y),f'shard {sh:02d}',fill='black');y+=20;can.paste(im,(0,y));y+=im.height
    can.save(out/'generated_50_identity_montage_v2.png')
print(json.dumps(agg,indent=2))
